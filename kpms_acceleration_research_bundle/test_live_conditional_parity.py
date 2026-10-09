#!/usr/bin/env python3
"""Additional checks requiring the user's installed JAX/Numba/KPMS environment.

Frozen-condition comparisons, no training or checkpoint writes. Compares full
Gaussian covariance against upstream draws, rather than trajectory correlation.
Run this in a fresh process; do not apply production monkeypatches first.
"""
import argparse
import copy
import hashlib
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from test_precompute_regressions import (
    dense_information, fixture, live_module,
)


class NoiseStream:
    def __init__(self, noise):
        self.noise = np.asarray(noise).ravel()
        self.start = 0

    def standard_normal(self, size=None, out=None):
        if out is None:
            raise AssertionError('Sampler noise API changed; update explicit noise injection.')
        end = self.start + out.size
        out[:] = self.noise[self.start:end].reshape(out.shape)
        self.start = end
        return out


RB = None
DRAWS = 4096


def optimized_gaussian(f, jitter):
    """Recover exact affine sampler map using zero and basis noise vectors."""
    import jax.random as jr
    args = {k:RB.jnp.asarray(v) for k,v in f.items()}
    RB.KPMSStateCache.clear()
    n = f['Y'].shape[0]*f['Y'].shape[1]*f['Ab'].shape[1]

    def sample(noise):
        with patch.object(RB.np.random,'default_rng',lambda seed: NoiseStream(noise)):
            out,repairs = RB.pure_numba_resample_continuous_stateseqs(
                jr.PRNGKey(81),**args,jitter=jitter,chunk_size=1,return_diagnostics=True)
        if repairs:
            raise AssertionError(f'{repairs} pivot repairs: covariance comparison invalid')
        return np.asarray(out).ravel()

    mean = sample(np.zeros(n))
    B = np.column_stack([sample(np.eye(n)[i])-mean for i in range(n)])
    return mean,B@B.T


class LiveConditionalTests(unittest.TestCase):
    def tearDown(self):
        RB.KPMSStateCache.clear()

    def test_emissions_and_hybrid_ffbs_batch_sizes(self):
        import jax.random as jr
        rng = np.random.default_rng(11)
        N,T,D,L,K = 7,17,2,3,3
        x = rng.normal(size=(N,T,D))
        mask = rng.uniform(size=(N,T))>.2
        Ab = rng.normal(0,.2,(K,D,L*D+1))
        q = rng.normal(size=(K,D,D)); Q = q@q.transpose(0,2,1)+np.eye(D)*.5
        pi = np.array([[.85,.10,.05],[.20,.75,.05],[.05,.15,.80]])
        expected = np.empty((N,T-L,K))
        for n in range(N):
            for t in range(L,T):
                for k in range(K):
                    delta = x[n,t]-Ab[k]@np.r_[x[n,t-L:t].ravel(),1.]
                    expected[n,t-L,k] = -.5*(D*np.log(2*np.pi)+np.linalg.slogdet(Q[k])[1]
                        +delta@np.linalg.solve(Q[k],delta))
        xd,md,ad,qd,pd = [RB.jnp.asarray(a) for a in [x,mask,Ab,Q,pi]]
        got = np.asarray(RB._jit_compute_discrete_emissions(xd,ad,qd,nlags=L,num_states=K))
        np.testing.assert_allclose(got,expected,rtol=2e-11,atol=2e-11)
        old = RB.CONFIG.use_numba_hmm
        try:
            RB.CONFIG.use_numba_hmm = True
            reference = None
            for bs in [7,1,2,3,4,8]:
                z = np.asarray(RB.fast_resample_discrete_stateseqs(
                    jr.PRNGKey(99),xd,md,ad,qd,pd,nlags=L,num_states=K,batch_size=bs))
                if reference is None: reference = z
                else: np.testing.assert_array_equal(z,reference)
        finally:
            RB.CONFIG.use_numba_hmm = old

    def test_upstream_conditional_mean_and_full_covariance(self):
        import jax
        import jax.random as jr
        from scipy.stats import chi2
        # Include boundaries, varying local noise, affine offsets and cross-lag terms.
        base = fixture(N=1,T=9,D=2,L=3)
        for mode in ['valid','trailing','internal']:
            for jitter in [0.,1e-3]:
                with self.subTest(mask=mode,jitter=jitter):
                    f = copy.deepcopy(base); f['mask'][:] = True
                    if mode == 'trailing': f['mask'][:,-2:] = False
                    if mode == 'internal': f['mask'][:,4:6] = False
                    mean,cov = optimized_gaussian(f,jitter)
                    gamma = np.asarray(RB.slds_alignment.center_embedding(f['Y'].shape[2]))
                    J,b = dense_information(f,gamma,jitter)
                    # Verify the exact production sampler before judging upstream draws.
                    np.testing.assert_allclose(mean,np.linalg.solve(J[0],b[0]),rtol=2e-10,atol=2e-10)
                    np.testing.assert_allclose(cov,np.linalg.inv(J[0]),rtol=2e-10,atol=2e-10)
                    args = {k:RB.jnp.asarray(v) for k,v in f.items()}

                    def draw(key):
                        return RB._orig_x(key,**args,jitter=jitter,parallel_message_passing=False)

                    draw_batch = jax.jit(jax.vmap(draw))
                    keys = jr.split(jr.PRNGKey(849),DRAWS)
                    chunks = [np.asarray(draw_batch(keys[i:i+256])) for i in range(0,DRAWS,256)]
                    samples = np.concatenate(chunks).reshape(DRAWS,-1)
                    self.assertTrue(np.isfinite(samples).all())
                    whitened = np.linalg.solve(np.linalg.cholesky(cov),(samples-mean).T).T
                    n_dim = mean.size
                    mean_z = np.sqrt(DRAWS)*whitened.mean(axis=0)
                    observed_cov = np.cov(whitened,rowvar=False,ddof=1)
                    diagonal = np.diag(observed_cov)
                    # Bonferroni across all six cases and all dimensions for variances.
                    tail = 1e-4/(2*6*n_dim)
                    lo,hi = chi2.ppf([tail,1-tail],DRAWS-1)/(DRAWS-1)
                    offdiag = observed_cov-np.diag(diagonal)
                    max_mean_z = np.max(np.abs(mean_z))
                    max_cross = np.max(np.abs(offdiag))*np.sqrt(DRAWS-1)
                    print(f'\nmask={mode} jitter={jitter:g}: max mean z={max_mean_z:.3f}; '
                          f'whitened variances=[{diagonal.min():.4f},{diagonal.max():.4f}]; '
                          f'max cross-cov z~={max_cross:.3f}',flush=True)
                    self.assertLess(max_mean_z,6.,'Conditional posterior mean differs from upstream')
                    self.assertTrue(np.all((diagonal>=lo)&(diagonal<=hi)),
                                    f'Posterior variance outside simultaneous [{lo},{hi}]')
                    self.assertLess(max_cross,6.,'Temporal/cross-coordinate covariance differs')


def main():
    global RB,DRAWS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rb',required=True,type=Path)
    parser.add_argument('--draws',type=int,default=4096)
    args = parser.parse_args()
    if args.draws < 2048:
        parser.error('Use at least 2048 independent frozen-condition draws.')
    DRAWS = args.draws
    RB = live_module(args.rb)
    import jax
    import numba
    import importlib.metadata
    print('Versions:',{name:importlib.metadata.version(name)
                       for name in ['jax','jaxlib','numba','jax-moseq','keypoint-moseq']},flush=True)
    print('Devices:',jax.devices(),'x64:',jax.config.jax_enable_x64,flush=True)
    print('Target SHA256:',hashlib.sha256(args.rb.read_bytes()).hexdigest(),flush=True)
    for fun in [RB._orig_x,RB.vanilla_slds_gibbs.resample_continuous_stateseqs]:
        path = Path(inspect.getsourcefile(inspect.unwrap(fun)))
        print('Upstream source:',path,'SHA256:',hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LiveConditionalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
