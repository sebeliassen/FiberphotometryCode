"""Pure and source-block regressions; no production arrays or workstation calls."""
from __future__ import annotations
import ast
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
_repo_source = ROOT / "syllable_generation/multiview_top_bot_experiments/04_reports_and_figures"
SOURCE = _repo_source if _repo_source.exists() else (ROOT / "patched_source")
sys.path.insert(0,str(SOURCE))
import viz_v2_common as vc


def load_functions(filename,names,namespace):
    tree=ast.parse((SOURCE/filename).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=nodes,type_ignores=[])
    import __future__
    exec(compile(ast.fix_missing_locations(module),str(SOURCE/filename),"exec",
                 flags=__future__.annotations.compiler_flag),namespace)
    return namespace


class V4Repairs(unittest.TestCase):
    def test_full_affine_composition_and_reflection(self):
        rng=np.random.default_rng(51)
        for A,b in [(np.array([[.37,-.02],[.01,.38]]),np.array([20.,70.])),
                    (np.array([[.35,.05],[-.02,.4]]),np.array([-10.,4.]))]:
            q=rng.uniform(0,972,(80,2));p=np.c_[972-q[:,0],q[:,1]]@A.T+b
            for h in [0.,np.pi/2,-np.pi/2,np.pi-.01,-2.1]:
                c=np.array([200.,170.])
                top,bot=vc.canonical_crop_matrices(c,h,A,b,972,160,144)
                got=np.c_[q,np.ones(len(q))]@bot.T
                expected=(160/144)*vc.egocentric_window(p[None],c,h)[0]+80
                np.testing.assert_allclose(got,expected,atol=1e-3)
                self.assertLess(np.linalg.det(bot[:,:2]),0)
                self.assertGreater(np.linalg.det(top[:,:2]),0)
                # A later horizontal flip would be a second reflection and fail parity.
                flipped=got.copy();flipped[:,0]=159-flipped[:,0]
                self.assertGreater(np.linalg.norm(flipped-expected),100)

    def test_fusion_geometry_requires_actual_recorded_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'session_tandem_21kp.h5';p.write_bytes(b'fixture')
            st=p.stat();meta={'kind':'fused_static_21kp','artifact':{'name':p.name,'size':st.st_size,'mtime_ns':st.st_mtime_ns},
                'config':{'affine_matrix':[[.37,0],[0,.37]],'affine_translation':[10,20],'arena_width':972,'epoch':'recorded_epoch'}}
            side=p.with_name(p.name+'.manifest.json');side.write_text(json.dumps(meta))
            geom=vc.fusion_geometry('session',d);self.assertEqual(geom['epoch'],'recorded_epoch')
            p.write_bytes(b'changed data')
            with self.assertRaises(ValueError):vc.fusion_geometry('session',d)

    def test_selector_and_renderer_context_boundary(self):
        self.assertTrue(vc.movie_context_ok(686,781,10,30))
        self.assertFalse(vc.movie_context_ok(686,781,60,120))
        self.assertTrue(vc.movie_context_ok(661,781,60,120))
        self.assertFalse(vc.movie_context_ok(59,781,60,120))

    def test_missing_fused_qc_does_not_pass(self):
        with self.assertRaises(RuntimeError):vc.require_fused_qc(None,'session')
        self.assertIsNone(vc.require_fused_qc(None,'session',skip=True))
        tree=ast.parse((SOURCE/'select_cohort_static_21kp_v2_syllable_manifest.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_fused')
        def missing(*args):raise FileNotFoundError('missing original tracks')
        fake=types.SimpleNamespace(load_fused_tracks=missing)
        ns={'vc':fake,'fused_cache':{},'FUSED_DIR':Path('/unused')}
        import __future__
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'get_fused','exec',flags=__future__.annotations.compiler_flag),ns)
        with self.assertRaises(RuntimeError):ns['get_fused']('session')

    def test_active_qc_catches_loss_after_scoring_window(self):
        mask=np.ones((100,21),bool);mask[40:,17:]=False
        self.assertTrue(vc.qc_decision(vc.summarize_window_qc(mask[:40]),vc.QCConfig(),60)[0])
        self.assertFalse(vc.qc_decision(vc.summarize_window_qc(mask),vc.QCConfig(),60)[0])

    def test_short_execution_does_not_extend_into_next_label(self):
        Y=np.zeros((40,21,2));Y[15:]=999
        got=vc.active_bout_window(Y,10,15)
        self.assertEqual(len(got),5);self.assertEqual(got.max(),0)
        with self.assertRaises(ValueError):vc.active_bout_window(Y,39,45)

    def test_context_difference_and_active_similarity_are_separate(self):
        a=np.zeros((40,21,2));b=a.copy();b[21:]=100;b[:10]=50
        shared,n=vc.shared_active_rms(a,b,11,32,10,30,range(21))
        self.assertEqual(n,11);self.assertEqual(shared,0)
        self.assertGreater(vc.rms_point_difference(a,b,range(21)),0)
        self.assertEqual(vc.score_execution_counts(11,10,30),{'prior_context_frames':10,'active_execution_frames':11,'following_context_frames':19,'score_window_frames':40})

    def test_manifest_rejects_different_arrays_or_debug_qc(self):
        class Cache:
            def __init__(self,suffix=''):self.suffix=suffix
            def get(self,path):return Path(path).name+self.suffix
        m={'schema_version':4,'input_hashes':vc.source_input_hashes(Cache()),'qc_config':{'applied':True}}
        vc.verify_manifest_inputs(m,Cache())
        with self.assertRaises(ValueError):vc.verify_manifest_inputs(m,Cache('changed'))
        m['qc_config']['applied']=False
        with self.assertRaises(ValueError):vc.verify_manifest_inputs(m,Cache())

    def test_gallery_rejects_flagged_blank_even_with_sidecar(self):
        m={'schema_version':4,'syllables':{'1':{'bouts':[{'bout_id':'a:10','is_blank':False}]}}}
        rec={'cache_key':{'schema_version':4,'manifest_syllable_sha256':vc.manifest_syllable_hash(m,1),'manifest_global_sha256':vc.manifest_global_hash(m)},
             'rendered_tiles':[{'bout_id':'a:10','is_blank':True,'flagged_invalid':True}]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'side.json';p.write_text(json.dumps(rec))
            self.assertFalse(vc.selection_provenance_matches(p,m,1))
            rec['rendered_tiles']=[{'bout_id':'a:10','is_blank':False}];p.write_text(json.dumps(rec))
            self.assertTrue(vc.selection_provenance_matches(p,m,1))

    def test_trajectory_failure_cannot_promote_neighbor_to_exemplar(self):
        p=SOURCE/'render_all_cohort_static_21kp_v2_seed101_trajectories.py';tree=ast.parse(p.read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='b' and ast.unparse(n.iter)=='manifest_bouts')
        ns={'np':np,'vc':vc,'manifest_bouts':[{'bout_id':'a:5','segment_name':'a','onset_segment_frame':5,'duration_frames':5}],
            'res_file':{'a':{'centroid':np.full((30,2),np.nan),'heading':np.zeros(30)}},
            'coords_dict':{'a':np.zeros((30,21,2))},'display_post':120,'extracted_trajs':[],'extracted_bouts':[]}
        with self.assertRaises(ValueError):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(p),'exec'),ns)
        self.assertEqual(ns['extracted_bouts'],[])

    def test_native_render_uses_true_frame_count_and_image_y_direction(self):
        ns={'np':np,'plt':plt,'vc':vc}
        load_functions('render_all_cohort_static_21kp_v2_seed101_trajectories.py',
            {'rasterize_figure','plot_trajectory_frame','remove_frame_artists','render_superimposed'},ns)
        traj=np.zeros((11,21,2));traj[:,:,0]=np.arange(21)[None]*2;traj[:,:,1]=np.arange(11)[:,None]
        fig,rasters=ns['render_superimposed'](traj,[traj[:2]],[(0,1)],np.tile([.2,.5,.8],(21,1)),'test','native frames',fps=60)
        self.assertEqual(len(rasters),11)
        self.assertGreater(fig.axes[0].get_ylim()[0],fig.axes[0].get_ylim()[1])
        plt.close(fig)

    def test_gif_cumulative_delay_matches_short_execution(self):
        ns={'np':np,'Image':Image}
        load_functions('render_all_cohort_static_21kp_v2_seed101_trajectories.py',{'save_gif'},ns)
        frames=[np.full((16,16,3),i*17,np.uint8) for i in range(11)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.gif';ns['save_gif'](frames,p,1/60)
            with Image.open(p) as im:
                delays=[]
                for i in range(im.n_frames):im.seek(i);delays.append(im.info['duration'])
                self.assertEqual(im.n_frames,11)
            self.assertLessEqual(abs(sum(delays)-11/60*1000),5)

    def test_production_probe_matches_fusion_float32_geometry(self):
        from probe_active_similarity import project_bot
        q=np.array([[[123.5,310.25],[945.,150.]]],np.float32)
        A=np.array([[.37,-.015],[.008,.38]],np.float32);b=np.array([18.,20.],np.float32)
        expected=np.stack([972-q[...,0],q[...,1]],axis=-1)@A.T+b
        np.testing.assert_array_equal(project_bot(q,A,b,972),expected)

    def test_production_probe_does_not_hide_fusion_identity_errors(self):
        from probe_active_similarity import identity_error
        q=np.zeros((3,21,2),np.float32);p=np.ones((3,21),np.float32)
        expected=q.copy();expected[1,17]=np.nan
        confidence=p.copy();confidence[1,17]=0
        result=identity_error(q,p,expected,confidence)
        self.assertEqual(result["coordinate_finite_mask_disagreements"],2)
        self.assertEqual(result["max_abs_likelihood_error"],1.)

    def test_forepaw_limited_policy_and_qc_invariance(self):
        # 1. Front-only loss rejects under strict QC, passes under dorsal_and_hind_supported_forepaw_limited
        mask_front_loss = np.ones((40, 21), bool)
        mask_front_loss[:, vc.FRONT_PAW_IDXS] = False  # front paws completely missing
        summary_front = vc.summarize_window_qc(mask_front_loss)

        cfg_strict = vc.QCConfig(policy="strict_all_paws")
        cfg_forepaw_ltd = vc.QCConfig(policy="dorsal_and_hind_supported_forepaw_limited")

        strict_pass, strict_reasons = vc.qc_decision(summary_front, cfg_strict, 60.0)
        self.assertFalse(strict_pass)
        self.assertIn("qc_paw_fraction", strict_reasons)

        ltd_pass, ltd_reasons = vc.qc_decision(summary_front, cfg_forepaw_ltd, 60.0)
        self.assertTrue(ltd_pass)
        self.assertEqual(len(ltd_reasons), 0)

        # 2. Hindpaw loss rejects under BOTH policies
        mask_hind_loss = np.ones((40, 21), bool)
        mask_hind_loss[:, vc.HIND_PAW_IDXS] = False
        summary_hind = vc.summarize_window_qc(mask_hind_loss)
        self.assertFalse(vc.qc_decision(summary_hind, cfg_strict, 60.0)[0])
        self.assertFalse(vc.qc_decision(summary_hind, cfg_forepaw_ltd, 60.0)[0])

        # 3. Dorsal loss rejects under BOTH policies
        mask_dorsal_loss = np.ones((40, 21), bool)
        mask_dorsal_loss[:, 0] = False  # nose missing
        summary_dorsal = vc.summarize_window_qc(mask_dorsal_loss)
        self.assertFalse(vc.qc_decision(summary_dorsal, cfg_strict, 60.0)[0])
        self.assertFalse(vc.qc_decision(summary_dorsal, cfg_forepaw_ltd, 60.0)[0])

        # 4. All-paw loss rejects under BOTH policies
        mask_all_paws_loss = np.ones((40, 21), bool)
        mask_all_paws_loss[:, vc.PAW_IDXS] = False
        summary_all_paws = vc.summarize_window_qc(mask_all_paws_loss)
        self.assertFalse(vc.qc_decision(summary_all_paws, cfg_strict, 60.0)[0])
        self.assertFalse(vc.qc_decision(summary_all_paws, cfg_forepaw_ltd, 60.0)[0])

        # 5. Gap thresholds respect FPS
        # 0.1s gap limit: 6 frames at 60 fps, 3 frames at 30 fps
        mask_gap5 = np.ones((40, 21), bool)
        mask_gap5[10:15, 0] = False  # 5-frame gap on nose
        summary_gap5 = vc.summarize_window_qc(mask_gap5)
        # At 60 fps (limit is 6), 5 frames should pass
        self.assertTrue(vc.qc_decision(summary_gap5, cfg_strict, 60.0)[0])
        # At 30 fps (limit is 3), 5 frames should fail
        self.assertFalse(vc.qc_decision(summary_gap5, cfg_strict, 30.0)[0])

        # 6. Repeated evaluations do not mutate QC summaries
        summary_copy = copy.deepcopy(summary_front)
        vc.qc_decision(summary_front, cfg_strict, 60.0)
        vc.qc_decision(summary_front, cfg_forepaw_ltd, 60.0)
        vc.qc_decision(summary_front, cfg_strict, 30.0)
        self.assertEqual(summary_front, summary_copy)


if __name__=='__main__':unittest.main(verbosity=2)
