#!/usr/bin/env python3
"""Targeted tests for the supplied resample_batching.py.

Source mode executes selected, unmodified function bodies with Numba decorators
removed and explicit NumPy substitutes for JAX/device plumbing. It tests Python
logic and numerical algebra, NOT JIT/GPU correctness or upstream equivalence.
Live mode imports the actual file and uses installed JAX/Numba/KPMS dependencies.
Neither mode trains a model, writes checkpoints, or installs monkeypatches.
"""
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import itertools
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np


class HostDevice:
    """Array conversion facade that is deliberately not a NumPy ndarray."""
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    @property
    def shape(self):
        return self.value.shape

    def __array__(self, dtype=None, copy=None):
        return np.array(self.value, dtype=dtype, copy=True)

    def __getitem__(self, item):
        return HostDevice(self.value[item])


class RecyclingIdentity:
    """Deterministic allocator model: IDs recycle only after object death.

    No two live objects receive the same ID. This replaces the module's lookup
    of id(), not Python's global builtin. A correct source-reference cache passes.
    """
    def __init__(self):
        self.refs = {}

    def __call__(self, obj):
        for token, ref in self.refs.items():
            if ref() is obj:
                return token
        free = next((k for k, ref in self.refs.items() if ref() is None), None)
        token = free if free is not None else len(self.refs) + 1
        self.refs[token] = weakref.ref(obj)
        return token


def embedding(k):
    # Orthonormal basis for the zero-centroid subspace; an explicit test fixture.
    return np.linalg.qr(np.eye(k)[:, :-1] - np.eye(k)[:, -1:])[0]


def source_module(path):
    names = {
        'BatchConfig', 'precompute_optimal_batch_config', 'KPMSStateCache',
        '_numba_sample_categorical', '_numba_ffbs_single_sequence',
        '_numba_ffbs_all_parallel', '_zero_alloc_accumulate_ar_stats',
        '_numba_banded_cholesky_inplace', '_numba_banded_forward_solve_inplace',
        '_numba_banded_backward_solve', '_numba_assemble_and_solve_all_parallel',
        '_numba_fused_rigid_transform', 'pure_numba_resample_continuous_stateseqs',
    }
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    found = {n.name for n in nodes}
    if found != names:
        raise RuntimeError(f'Source interface changed; inspect missing names: {names-found}')
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    import dataclasses
    from typing import Optional
    rb = types.ModuleType('_kpms_source_under_test')
    sys.modules[rb.__name__] = rb
    rb.__dict__.update(
        np=np, prange=range, dataclass=dataclasses.dataclass, Optional=Optional,
        jnp=np, jr=types.SimpleNamespace(randint=lambda seed, *a: int(seed)),
        jax=types.SimpleNamespace(devices=lambda: []),
        slds_alignment=types.SimpleNamespace(center_embedding=embedding),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), rb.__dict__)
    rb.CONFIG = rb.BatchConfig()
    return rb


def live_module(path):
    import jax
    jax.config.update('jax_enable_x64', True)
    # Some project imports resolve relative to the KPMS root.
    sys.path.insert(0, str(path.parent.parent))
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('_kpms_live_under_test', path)
    rb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rb
    spec.loader.exec_module(rb)
    return rb


def fixture(N=7, T=9, D=2, L=3, K=3, keypoints=4):
    rng = np.random.default_rng(720)
    Ab = rng.normal(0, .13, (K, D, L*D+1))
    Ab[:, :, -1] += np.arange(K)[:, None]*.4
    a = rng.normal(size=(K, D, D))
    Q = a @ a.transpose(0, 2, 1) + .6*np.eye(D)
    mask = np.ones((N, T), dtype=bool)
    mask[0, -2:] = False
    if N > 1:
        mask[1, L+1] = False
    return dict(
        Y=rng.normal(size=(N,T,keypoints,2)), mask=mask,
        v=rng.normal(size=(N,T,2)), h=rng.uniform(-2,2,(N,T)),
        s=rng.uniform(.2,3,(N,T,keypoints)),
        z=rng.integers(K,size=(N,T-L),dtype=np.int32),
        Cd=rng.normal(size=((keypoints-1)*2,D+1)),
        sigmasq=np.geomspace(.2,2,keypoints), Ab=Ab, Q=Q,
    )


def dense_information(f, gamma, jitter):
    """Independent global residual-design oracle for the implemented model.

    Each AR factor is built directly into a global design matrix. No precomputed
    blocks, band assembly, or production lag helper are used. Masked factors,
    0.1 initial precision, 1e-8 observation floor, and precision jitter deliberately
    match the supplied implementation; this alone does not establish upstream parity.
    """
    N,T,k,d = f['Y'].shape
    D = f['Ab'].shape[1]
    L = (f['Ab'].shape[2]-1)//D
    Cd = np.kron(gamma,np.eye(d)) @ f['Cd']
    C, offset = Cd[:,:-1], Cd[:,-1]
    matrices, vectors = [], []
    for n in range(N):
        J = np.eye(T*D)*jitter
        J[:L*D,:L*D] += .1*np.eye(L*D)
        b = np.zeros(T*D)
        for t in range(T):
            h = f['h'][n,t]
            R = np.array([[np.cos(h),np.sin(h)],[-np.sin(h),np.cos(h)]])
            y = ((f['Y'][n,t]-f['v'][n,t]) @ R.T).ravel()-offset
            inv_R = (1/(np.repeat(f['sigmasq']*f['s'][n,t],2)+1e-8)
                     if f['mask'][n,t] else np.full(k*d,.1))
            F = np.zeros((k*d,T*D)); F[:,t*D:(t+1)*D] = C
            J += F.T @ (inv_R[:,None]*F)
            b += F.T @ (inv_R*y)
            if t < L:
                continue
            F = np.zeros((D,T*D)); F[:,t*D:(t+1)*D] = np.eye(D)
            if f['mask'][n,t]:
                z = f['z'][n,t-L]
                for lag in range(L):
                    F[:,(t-L+lag)*D:(t-L+lag+1)*D] = -f['Ab'][z,:,lag*D:(lag+1)*D]
                Q, bias = f['Q'][z], f['Ab'][z,:,-1]
            else:
                F[:,(t-1)*D:t*D] = -np.eye(D)
                Q, bias = 10*np.eye(D), np.zeros(D)
            J += F.T @ np.linalg.solve(Q,F)
            b += F.T @ np.linalg.solve(Q,bias)
        matrices.append(J); vectors.append(b)
    return np.array(matrices),np.array(vectors)


def dense_samples(f, gamma, seed, jitter):
    J,b = dense_information(f,gamma,jitter)
    noise = np.random.default_rng(seed).standard_normal(b.shape)
    result = []
    for Ji,bi,eta in zip(J,b,noise):
        chol = np.linalg.cholesky(Ji)
        result.append(np.linalg.solve(Ji,bi)+np.linalg.solve(chol.T,eta))
    return np.array(result).reshape(f['Y'].shape[:2]+(f['Ab'].shape[1],))


def exhaustive_hmm(ll, pi, mask, uniforms):
    """Enumerate all paths, then condition on sampled suffixes independently."""
    T,K = ll.shape
    paths = np.array(list(itertools.product(range(K),repeat=T)))
    logp = np.full(len(paths),-np.log(K))
    for t in range(T):
        if mask[t]: logp += ll[t,paths[:,t]]
        if t: logp += np.log(pi[paths[:,t-1],paths[:,t]])
    weights = np.exp(logp-logp.max())
    selected = np.zeros(T,dtype=np.int32)
    compatible = np.ones(len(paths),dtype=bool)
    for t in range(T-1,-1,-1):
        p = np.bincount(paths[compatible,t],weights=weights[compatible],minlength=K)
        p /= p.sum()
        selected[t] = min(np.searchsorted(np.cumsum(p),uniforms[t],side='right'),K-1)
        compatible &= paths[:,t] == selected[t]
    return selected


RB = None
LIVE = False


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        RB.KPMSStateCache.clear()
        self.old_config = copy.deepcopy(vars(RB.CONFIG))

    def tearDown(self):
        RB.KPMSStateCache.clear()
        vars(RB.CONFIG).update(self.old_config)

    def seed(self, value=1729):
        return RB.jr.PRNGKey(value) if LIVE else value

    def arrays(self, f):
        return {k:(RB.jnp.asarray(v) if LIVE else HostDevice(v)) for k,v in f.items()}

    def call(self, f, chunk=3, jitter=0):
        x,repairs = RB.pure_numba_resample_continuous_stateseqs(
            self.seed(), **f, chunk_size=chunk, jitter=jitter, return_diagnostics=True)
        self.assertEqual(repairs,0,'A repaired factorization cannot silently pass parity.')
        self.assertTrue(np.isfinite(np.asarray(x)).all())
        return np.asarray(x)

    def test_01_cache_expired_source_identity_reuse(self):
        for name in ['x','z','mask','Y']:
            for use_update in ([False,True] if name in ['x','z'] else [False]):
                with self.subTest(field=name,producer_update=use_update):
                    RB.KPMSStateCache.clear()
                    allocator = RecyclingIdentity()
                    with patch.dict(RB.__dict__, {'id':allocator}):
                        old = HostDevice([0,0])
                        get = getattr(RB.KPMSStateCache,'get_'+name)
                        if use_update:
                            getattr(RB.KPMSStateCache,'update_'+name)(np.array([0,0]),old)
                        else:
                            get(old)
                        del old
                        gc.collect()
                        new = HostDevice([1,1])
                        np.testing.assert_array_equal(get(new),np.asarray(new),
                            err_msg=f'Stale {name} returned after expired object ID reuse')

    def test_02_numpy_mutation_is_visible(self):
        for name in ['x','z','mask','Y']:
            with self.subTest(field=name):
                get = getattr(RB.KPMSStateCache,'get_'+name)
                value = np.array([0,0]); get(value); value[:] = 1
                np.testing.assert_array_equal(get(value),value)

    def test_03_continuous_sample_matches_independent_dense_oracle(self):
        for L,D in [(1,1),(2,2),(3,2)]:
            for jitter in [0.,1e-3]:
                with self.subTest(lags=L,dim=D,jitter=jitter):
                    RB.KPMSStateCache.clear()  # Isolate algebra from the lifecycle test.
                    f = fixture(N=3,L=L,D=D)
                    gamma = np.asarray(RB.slds_alignment.center_embedding(f['Y'].shape[2]))
                    seed_int = int(RB.jr.randint(self.seed(),(),0,2**31-1))
                    expected = dense_samples(f,gamma,seed_int,jitter)
                    actual = self.call(self.arrays(f),jitter=jitter)
                    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)

    def test_04_full_continuous_chunk_reuse_and_partial_chunks(self):
        f = self.arrays(fixture())
        expected = self.call(f,chunk=7)
        for chunk in [1,2,3,4,8]:
            with self.subTest(chunk=chunk):
                np.testing.assert_allclose(self.call(f,chunk),expected,rtol=2e-11,atol=2e-11)

    def test_05_dependency_changes_match_forced_fresh_evaluation(self):
        base = fixture(N=3)
        baseline = self.call(self.arrays(base))
        for field in ['Ab','Q','z','Cd','sigmasq','s','h','v','Y','mask']:
            with self.subTest(dependency=field):
                RB.KPMSStateCache.clear()
                live_base = self.arrays(base)
                self.call(live_base)
                changed = copy.deepcopy(base)
                if field == 'mask': changed[field][0,4] = ~changed[field][0,4]
                elif field == 'z': changed[field] = (changed[field]+1)%base['Ab'].shape[0]
                elif field in ['Q','sigmasq','s']: changed[field] *= 1.7
                else: changed[field] += .4
                # Keep unchanged inputs alive; change only one object at this point.
                warm_args = dict(live_base)
                warm_args[field] = self.arrays({field:changed[field]})[field]
                warm = self.call(warm_args)
                RB.KPMSStateCache.clear()
                cold = self.call(self.arrays(changed))
                self.assertGreater(np.max(np.abs(cold-baseline)),1e-7,
                                   'Fixture did not exercise the changed dependency')
                np.testing.assert_allclose(warm,cold,rtol=2e-11,atol=2e-11)

    def test_06_banded_noise_map_has_correct_full_covariance(self):
        # Exact basis-noise test: no noisy correlation proxy or Monte Carlo needed.
        rng = np.random.default_rng(4)
        n,bw = 13,4
        L = np.diag(rng.uniform(1,2,n))
        for row in range(n):
            for col in range(max(0,row-bw),row): L[row,col] = rng.normal(0,.2)
        J = L@L.T
        band = np.zeros((bw+1,n))
        for col in range(n):
            for row in range(col,min(n,col+bw+1)): band[row-col,col] = J[row,col]
        self.assertEqual(RB._numba_banded_cholesky_inplace(band,bw,n),0)
        B = np.zeros((n,n))
        for i in range(n):
            eta = np.eye(n)[i]
            RB._numba_banded_backward_solve(band,bw,n,eta,B[:,i])
        np.testing.assert_allclose(B@B.T,np.linalg.inv(J),rtol=1e-11,atol=1e-11)
        g = rng.normal(size=n); w = g.copy(); x = np.zeros(n)
        RB._numba_banded_forward_solve_inplace(band,bw,n,w)
        RB._numba_banded_backward_solve(band,bw,n,w,x)
        np.testing.assert_allclose(J@x,g,rtol=1e-11,atol=1e-11)

    def test_07_ffbs_matches_enumerated_joint_not_just_state_counts(self):
        ll = np.array([[.3,-.8],[1.,-.2],[-.3,.9],[.7,.2],[.1,-1.]])
        pi = np.array([[.93,.07],[.31,.69]])
        rng = np.random.default_rng(120)
        for mask in [np.ones(5,bool),np.array([0,1,0,1,0],bool),np.zeros(5,bool)]:
            for draw in range(30):
                with self.subTest(mask=mask.tolist(),draw=draw):
                    u = rng.uniform(size=5); out = np.empty(5,np.int32); P = np.empty_like(ll)
                    RB._numba_ffbs_single_sequence(ll,np.ascontiguousarray(pi.T),mask,u,out,P)
                    np.testing.assert_array_equal(out,exhaustive_hmm(ll,pi,mask,u))
                    np.testing.assert_allclose(P.sum(axis=1),1,rtol=1e-13,atol=1e-13)

    def test_08_ar_statistics_use_new_x_z_mask_and_correct_lags(self):
        rng = np.random.default_rng(88)
        N,T,D,L,K = 3,11,2,3,4
        x = rng.normal(size=(N,T,D)); z = rng.integers(K,size=(N,T-L),dtype=np.int32)
        mask = rng.uniform(size=(N,T))>.2
        actual = RB._zero_alloc_accumulate_ar_stats(x,z,mask,L,K)
        cnt = np.zeros(K); ii = np.zeros((K,L*D+1,L*D+1))
        oi = np.zeros((K,D,L*D+1)); oo = np.zeros((K,D,D))
        for n in range(N):
            for t in range(L,T):
                if not mask[n,t]: continue
                k = z[n,t-L]; a = np.r_[x[n,t-L:t].ravel(),1.]; y = x[n,t]
                cnt[k] += 1; ii[k] += np.outer(a,a); oi[k] += np.outer(y,a); oo[k] += np.outer(y,y)
        for got,want in zip(actual,[cnt,ii,oi,oo]):
            np.testing.assert_allclose(got,want,rtol=2e-12,atol=2e-12)

    def test_09_batch_sizer_changes_only_config_and_preserves_sample(self):
        f = fixture(); before = copy.deepcopy(f)
        model = {'params':{'Ab':f['Ab'],'pi':np.eye(f['Ab'].shape[0])}}
        configs = []
        expected = None
        for budget in [1e-7,.001,1.]:
            RB.precompute_optimal_batch_config({'Y':f['Y']},model,
                gpu_budget_gb=budget,cpu_budget_gb=budget,verbose=False)
            configs.append(tuple(vars(RB.CONFIG).values()))
            for name in ['discrete_bs','scales_bs','heading_bs','gpu_loc_bs','continuous_chunk_size']:
                self.assertTrue(1 <= getattr(RB.CONFIG,name) <= len(f['Y']))
            x = self.call(self.arrays(f),chunk=RB.CONFIG.continuous_chunk_size)
            if expected is None: expected = x
            else: np.testing.assert_allclose(x,expected,rtol=2e-11,atol=2e-11)
        self.assertGreater(len(set(configs)),1)
        for key in f: np.testing.assert_array_equal(f[key],before[key])

    def test_10_continuous_inputs_are_not_mutated(self):
        f = fixture(N=3); before = copy.deepcopy(f)
        self.call(f,chunk=2)
        for key in f: np.testing.assert_array_equal(f[key],before[key])


def main():
    global RB,LIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rb',required=True,type=Path)
    parser.add_argument('--backend',choices=['source','live'],default='source')
    args = parser.parse_args()
    LIVE = args.backend == 'live'
    print(f'Backend: {args.backend}; target: {args.rb.resolve()}',flush=True)
    print(f'SHA256: {hashlib.sha256(args.rb.read_bytes()).hexdigest()}',flush=True)
    RB = live_module(args.rb) if LIVE else source_module(args.rb)
    if LIVE:
        import jax
        print(f'JAX {jax.__version__}; devices={jax.devices()}; x64={jax.config.jax_enable_x64}',flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrecomputeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
