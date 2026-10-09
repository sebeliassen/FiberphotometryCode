import os
import numpy as np
import keypoint_moseq as kpms
import jax

# Prevent JAX from pre-allocating GPU memory during setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)

def generate_synthetic_environment(
    N_seqs=200,      
    Frames=10030,    # Matches your log output
    K_bodyparts=17,  
    D=2,             
    Latent=10,       
    nlags=3,         
    seed=0
):
    """
    Generates a valid Keypoint-MoSeq environment (Data, PCA, Config).
    """
    print(f"\n[SYNTHETIC SETUP] Generating {N_seqs} videos x {Frames} frames...")
    np.random.seed(seed)
    
    # 1. Generate Raw Keypoints
    coordinates = {}
    confidences = {}
    
    for i in range(N_seqs):
        coords = np.cumsum(np.random.randn(Frames, K_bodyparts, D), axis=0)
        confs = np.ones((Frames, K_bodyparts))
        coordinates[f"sim_video_{i}"] = coords
        confidences[f"sim_video_{i}"] = confs

    # 2. Define Anatomy
    bodyparts = [f"bp_{i}" for i in range(K_bodyparts)]
    anterior_bp = bodyparts[0] 
    posterior_bp = bodyparts[1]

    # 3. Format Data
    print("[SYNTHETIC SETUP] Formatting data via kpms...")
    data, metadata = kpms.format_data(
        coordinates, confidences,
        use_bodyparts=bodyparts,
        anterior_bodyparts=[anterior_bp],
        posterior_bodyparts=[posterior_bp]
    )
    
    # 4. Fit PCA
    print("[SYNTHETIC SETUP] Fitting PCA...")
    subset_data = {'Y': data['Y'][:5], 'mask': data['mask'][:5], 'conf': data['conf'][:5]}
    
    pca_config = {
        'anterior_idxs': [0],
        'posterior_idxs': [1],
        'conf_threshold': 0.5
    }
    
    pca = kpms.fit_pca(**subset_data, **pca_config, latent_dim=Latent)

    # 5. Generate Full Config
    print("[SYNTHETIC SETUP] Building robust config...")
    
    input_dim = Latent * nlags + 1
    S_0_scale = 0.01
    K_0_scale = 10.0
    
    config = {
        'anterior_idxs': [0],
        'posterior_idxs': [1],
        'conf_threshold': 0.5,
        'sls_sigma': 1.0,
        'error_estimator': {'intercept': 0.8, 'slope': -0.5},
        'obs_hypparams': {'nu_s': 5, 'nu_sigma': 1e5, 'sigmasq_0': 0.1, 'sigmasq_C': 0.1},
        'trans_hypparams': {'alpha': 5.7, 'gamma': 1e3, 'kappa': 1e6, 'num_states': 100},
        'cen_hypparams': {'sigmasq_loc': 1.6},
        'ar_hypparams': {
            'latent_dim': Latent,
            'nlags': nlags,
            'num_states': 100,
            
            # --- FIX: Provide BOTH Scalars (for checks) AND Matrices (for math) ---
            'S_0_scale': S_0_scale,
            'K_0_scale': K_0_scale,

            'S_0': S_0_scale * np.eye(Latent),
            'M_0': np.zeros((Latent, input_dim)),
            'K_0': K_0_scale * np.eye(input_dim),
            'nu_0': Latent + 2
        }
    }
    
    print("[SYNTHETIC SETUP] Ready.\n")
    return data, pca, config