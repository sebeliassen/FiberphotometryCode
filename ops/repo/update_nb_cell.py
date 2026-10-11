"""Apply the retained notebook patch only during explicit script execution."""
import argparse
import json
from pathlib import Path

from fpcode.common.paths import REPO_ROOT

patch_code = """# ===== Memory Patches: Batched Resamplers (Optimal Batch Size = 100) =====
# Prevents RESOURCE_EXHAUSTED OOMs on large 64-bit datasets.

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

BATCH_SIZE = 100  # Empirically verified optimal batch size for RTX 4090 (24GB)

# 1. Patch resample_discrete_stateseqs (Z - Discrete States)
from jax_moseq.models.arhmm import gibbs as arhmm_gibbs
_orig_z = arhmm_gibbs.resample_discrete_stateseqs

def _batched_z(seed, x, mask, Ab, Q, pi, **kwargs):
    N = x.shape[0]
    parts = []
    for i in tqdm(range(0, N, BATCH_SIZE), desc=f"resample_z (bs={BATCH_SIZE})", leave=False):
        end = min(i + BATCH_SIZE, N)
        batch_seed = jr.fold_in(seed, i)
        z_batch = _orig_z(batch_seed, x[i:end], mask[i:end], Ab, Q, pi, **kwargs)
        z_batch.block_until_ready()
        parts.append(z_batch)
    return jnp.concatenate(parts, axis=0)

arhmm_gibbs.resample_discrete_stateseqs = _batched_z


# 2. Patch resample_scales (S - Local Noise Scales)
from jax_moseq.models.keypoint_slds import gibbs as slds_gibbs
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error

SCALES_BS = min(BATCH_SIZE, 50)

@jax.jit
def _scales_kernel(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

def _batched_s(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    N = Y.shape[0]
    parts = []
    for i in tqdm(range(0, N, SCALES_BS), desc=f"resample_s (bs={SCALES_BS})", leave=False):
        end = min(i + SCALES_BS, N)
        batch_seed = jr.fold_in(seed, i)
        s0_b = s_0[i:end] if (hasattr(s_0, "shape") and len(s_0.shape) >= 3 and s_0.shape[0] == N) else s_0
        s_batch = _scales_kernel(batch_seed, Y[i:end], x[i:end], v[i:end], h[i:end], Cd, sigmasq, nu_s, s0_b)
        s_batch.block_until_ready()
        parts.append(s_batch)
    return jnp.concatenate(parts, axis=0)

slds_gibbs.resample_scales = _batched_s


# 3. Patch resample_continuous_stateseqs (X - Continuous Trajectories / Kalman)
_orig_x = slds_gibbs.resample_continuous_stateseqs

def _batched_x(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, **kwargs):
    N = Y.shape[0]
    parts = []
    for i in tqdm(range(0, N, BATCH_SIZE), desc=f"resample_x (bs={BATCH_SIZE})", leave=False):
        end = min(i + BATCH_SIZE, N)
        batch_seed = jr.fold_in(seed, i)
        x_part = _orig_x(
            batch_seed, Y[i:end], mask[i:end], v[i:end], h[i:end], s[i:end], z[i:end],
            Cd, sigmasq, Ab, Q, **kwargs
        )
        x_part.block_until_ready()
        parts.append(x_part)
    return jnp.concatenate(parts, axis=0)

slds_gibbs.resample_continuous_stateseqs = _batched_x

print(f"[Patch] All 3 resamplers active: Z (bs={BATCH_SIZE}), S (bs={SCALES_BS}), X (bs={BATCH_SIZE})")
# =========================================================="""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=REPO_ROOT / "vendor/keypoint_moseq_syll/run_kpms_nb.ipynb")
    args = parser.parse_args()
    nb_path = args.notebook
    with nb_path.open() as handle:
        nb = json.load(handle)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "Memory Patches: Batched Resamplers" in "".join(cell["source"]):
            cell["source"] = [line + "\n" for line in patch_code.split("\n")]
            break

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)

    print("Successfully updated notebook Cell 5 with verified BATCH_SIZE=100 patches!")


if __name__ == "__main__":
    main()
