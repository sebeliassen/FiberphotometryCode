"""
Regenerate syllable sub-state diagnostic plots using the updated
KinematicSyllableClustered (with log1p pre-transform).

Outputs mirror the original rich_tokens/ plots:
  - syllable_<id>_substates.png  (PCA scatter coloured by substate)
  - vocabulary_expansion_summary.png (bar chart of substates per syllable)
"""
import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.fusion.clustering import (
    _LOG1P_COLS,
    KinematicSyllableClustered,
    extract_bouts_from_session,
)
from analysis.ulrik_kinematics.lib import loader
from sklearn.decomposition import PCA

OUT_DIR = os.path.join(
    ROOT_DIR,
    "syllable_generation/analysis/language_models/output/rich_tokens"
)
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_KEYS = [
    "duration_frames", "mean_velocity", "mean_angular_velocity",
    "mean_acceleration", "mean_dist_from_center",
]
PALETTE = ["#7570b3", "#1b9e77", "#d95f02"]  # matches original colour scheme

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading sessions...")
sessions = []
try:
    s = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase="pre")
    sessions.extend(s)
    print(f"  AMPH pre: {len(s)}")
except Exception as e:
    print(f"  AMPH failed: {e}")
try:
    s = loader.load_dataset("ifdn", base_dir=ROOT_DIR)
    sessions.extend(s)
    print(f"  IFDN: {len(s)}")
except Exception as e:
    print(f"  IFDN failed: {e}")

print(f"Total sessions: {len(sessions)}")

# ---------------------------------------------------------------------------
# Fit updated clusterer
# ---------------------------------------------------------------------------
print("Fitting KinematicSyllableClustered (with log1p)...")
clusterer = KinematicSyllableClustered(n_clusters=3, min_bouts=10, use_gpu=False)
clusterer.fit(sessions)
print(f"  Fitted. Vocabulary size: {clusterer.next_token_id - 1} rich tokens")

# ---------------------------------------------------------------------------
# Collect all bouts with cluster assignments
# ---------------------------------------------------------------------------
all_bouts = []
for s in sessions:
    all_bouts.extend(extract_bouts_from_session(s))

syllable_bouts = defaultdict(list)
for b in all_bouts:
    syllable_bouts[b["syllable"]].append(b)

# Assign substates using fitted model

def get_labels_for_syllable(syll_id):
    bouts = syllable_bouts.get(syll_id, [])
    if not bouts:
        return None, None
    scaler, kmeans = clusterer.syllable_models.get(syll_id, (None, None))
    if scaler is None or kmeans is None:
        return None, None
    X = np.array([[b[k] for k in FEATURE_KEYS] for b in bouts], dtype=np.float32)
    X[:, _LOG1P_COLS] = np.log1p(X[:, _LOG1P_COLS])
    X_sc = scaler.transform(X)
    labels = kmeans.predict(X_sc)
    return X_sc, labels

# ---------------------------------------------------------------------------
# Plot 1: syllable_X_substates.png  (re-generate for ALL syllables)
# ---------------------------------------------------------------------------
print("\nGenerating per-syllable substate PCA plots...")
plotted = 0
for syll_id in sorted(syllable_bouts.keys()):
    X_sc, labels = get_labels_for_syllable(syll_id)
    if X_sc is None:
        continue

    n_bouts = len(labels)
    k = len(set(labels))

    # PCA to 2D for visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(10, 8))
    for ci in range(k):
        mask = labels == ci
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.55,
            c=PALETTE[ci % len(PALETTE)],
            label=str(ci),
        )
    ax.set_title(f"Syllable {syll_id} Sub-State Clustering (Bouts = {n_bouts})", fontsize=14)
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    legend = ax.legend(title="Substate Index", fontsize=10, title_fontsize=10)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"syllable_{syll_id}_substates.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plotted += 1

print(f"  Saved {plotted} substate plots to {OUT_DIR}/")

# ---------------------------------------------------------------------------
# Plot 2: vocabulary_expansion_summary.png
# ---------------------------------------------------------------------------
print("Generating vocabulary expansion summary...")
syll_ids = sorted(syllable_bouts.keys())
n_substates = []
for syll_id in syll_ids:
    scaler, kmeans = clusterer.syllable_models.get(syll_id, (None, None))
    if kmeans is None:
        n_substates.append(1)
    else:
        try:
            n_substates.append(int(kmeans.n_clusters))
        except Exception:
            n_substates.append(1)

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(syll_ids, n_substates, color="#5bafd6", edgecolor="#3d7fa6", linewidth=0.5)
ax.set_xlabel("Base Syllable ID", fontsize=12)
ax.set_ylabel("Number of Sub-States (Rich Tokens)", fontsize=12)
ax.set_title("Kinematic Sub-State Vocabulary Expansion per Syllable", fontsize=14)
ax.set_xticks([s for s in syll_ids if s % 5 == 0])
ax.set_ylim(0, max(n_substates) + 0.5)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "vocabulary_expansion_summary.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")

print("\nDone.")
