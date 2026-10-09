import os
import sys

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import pickle

sys.path.append('/home/fsp585/Desktop/FiberphotometryCode')
from syllable_generation.syllabizers.keypoint_moseq_syll.kpms_pipeline import run_pipeline

SESSIONS_PKL = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl'
with open(SESSIONS_PKL, 'rb') as f:
    sessions = pickle.load(f)

run_pipeline(
    sessions=sessions,
    project_dir='/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project',
    model_name='test_flip_logic',
    num_ar_iters=1,
    total_iters=1,
    flipping_mode='optimized',
    subset_fraction=0.01,
)
