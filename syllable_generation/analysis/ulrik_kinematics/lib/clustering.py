import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from sklearn.preprocessing import StandardScaler
import umap

from analysis.ulrik_kinematics import config
from analysis.ulrik_kinematics.lib import metrics
from analysis.ulrik_kinematics.lib import motifs

def extract_temporal_delta(session):
    import copy
    df = session.all_data_df
    mid_idx = len(df) // 2
    
    # Needs a mock session to reuse standard metrics logic
    s_early = copy.copy(session)
    s_early.all_data_df = df.iloc[:mid_idx].copy()
    
    s_late = copy.copy(session)
    s_late.all_data_df = df.iloc[mid_idx:].copy()
    
    kin_e = metrics.calculate_kinematics(s_early)
    kin_l = metrics.calculate_kinematics(s_late)
    
    ent_e = metrics.calculate_entropy_metrics(s_early)
    ent_l = metrics.calculate_entropy_metrics(s_late)
    
    return {
        "Delta_Mean_Velocity": kin_l.get("mean_velocity", 0.0) - kin_e.get("mean_velocity", 0.0),
        "Delta_Sequence_Entropy": ent_l.get("entropy_transition", 0.0) - ent_e.get("entropy_transition", 0.0)
    }

def extract_features(session, target_motifs, mapping=None):
    res = {}
    
    # ---- 1. BASE PHYSICS (Always available baseline) ----
    kin = metrics.calculate_kinematics(session)
    res["phys_Mean_Velocity"] = kin.get("mean_velocity", 0.0)
    res["phys_Mean_Acceleration"] = kin.get("mean_acceleration", 0.0)
    
    ent = metrics.calculate_entropy_metrics(session)
    res["phys_Sequence_Entropy"] = ent.get("entropy_transition", 0.0)
    
    # ---- DOMAIN A: POSTURAL COVARIANCE ----
    if hasattr(session, "syllables") and isinstance(session.syllables, pd.DataFrame):
        latent_cols = sorted([c for c in session.syllables.columns if c.startswith("latent_state_")])
        if latent_cols:
            df_lat = session.syllables[latent_cols].dropna()
            if len(df_lat) > 0:
                cov_mat = np.cov(df_lat.values, rowvar=False)
                # Extract upper triangle including diagonal
                for i in range(len(latent_cols)):
                    for j in range(i, len(latent_cols)):
                        # If i==j it is variance, otherwise covariance
                        res[f"cov_{i}_{j}"] = cov_mat[i, j]
                        
    # ---- DOMAIN B: BEHAVIORAL STRATEGY ----
    counts = motifs.extract_ngrams(session, n=config.MOTIF_LENGTH, mapping=mapping)
    total_tri = counts.sum() if not counts.empty else 0
    for m in target_motifs:
        prob = counts.get(m, 0) / total_tri if total_tri > 0 else 0
        res[f"strat_Motif_{m}"] = prob
        
    # ---- DOMAIN C: TEMPORAL HABITUATION ----
    deltas = extract_temporal_delta(session)
    for k, v in deltas.items():
        res[f"temp_{k}"] = v
                
    return res

def build_feature_matrix(sessions, target_motifs, mapping=None):
    rows = []
    ids = []
    for s in sessions:
        if "syllable" not in s.all_data_df.columns: continue
        feat = extract_features(s, target_motifs, mapping)
        rows.append(feat)
        
        # Clean drug naming
        d = s.metadata.get("drug")
        if d is None or not isinstance(d, str): d = "N/A"
        elif d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"
        else: d = "N/A"
        
        ids.append({
            "Session": s.session_id,
            "Animal": s.metadata.get("animal_id", "UNK"),
            "Genotype": s.metadata.get("genotype", "UNK"),
            "Drug": d
        })
    df_feat = pd.DataFrame(rows)
    df_meta = pd.DataFrame(ids)
    return df_feat, df_meta

def verify_clustering_module(sessions):
    """
    Implements the 2-Step Invariance / Perurbation Defense for dimensionality clustering.
    Ensures deterministic projection and topological distance validity.
    """
    print("\n" + "="*50)
    print(" VERIFICATION: TOPOLOGICAL CLUSTERING & EJECTION")
    print("="*50)
    
    mapping = motifs.get_label_mapping(sessions, style="standard")
    
    print("[1/4] Establishing UMAP baseline from random seed sessions...")
    # Grab 12 sessions so UMAP has enough geometry to build a mesh
    base_sess = sessions[:12]
    s_target = base_sess[0]
    
    # Clone 1: The Identical Twin (Invariance Check)
    print("[2/4] Injecting identical mathematical clone (Invariance Check)...")
    s_clone = copy.deepcopy(s_target)
    s_clone.session_id = "IDENTICAL_CLONE"
    
    # Clone 2: The Frankenstein (Perturbation Ejection Check)
    print("[3/4] Injecting sequence-shuffled Frankenstein (Perturbation Check)...")
    s_pert = copy.deepcopy(s_target)
    s_pert.session_id = "FRANKENSTEIN_PERT"
    # Destroying the temporal sequences but keeping physics intact
    np.random.seed(42)
    np.random.shuffle(s_pert.all_data_df["syllable"].values)
    
    test_sessions = base_sess + [s_clone, s_pert]
    
    # Quickly grab the Top 10 intentional-style motives inside the test chunk
    trigrams_all = pd.Series(dtype=int)
    for s in test_sessions:
        t = motifs.extract_ngrams(s, n=3, mapping=mapping)
        if not t.empty: trigrams_all = trigrams_all.add(t, fill_value=0)
    top_10 = trigrams_all.sort_values(ascending=False).head(10).index.tolist()
    
    print("[4/4] Executing vector extraction and wrapping UMAP projection...")
    df_feat, df_meta = build_feature_matrix(test_sessions, top_10, mapping=mapping)
    
    # Normalize features otherwise velocity dominates
    X = StandardScaler().fit_transform(df_feat.fillna(0))
    
    # Execute non-linear crushing algorithm
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    proj = reducer.fit_transform(X)
    
    df_meta["UMAP_X"] = proj[:, 0]
    df_meta["UMAP_Y"] = proj[:, 1]
    
    # Math constraints
    target_idx = df_meta[df_meta["Session"] == s_target.session_id].index[0]
    clone_idx = df_meta[df_meta["Session"] == "IDENTICAL_CLONE"].index[0]
    pert_idx = df_meta[df_meta["Session"] == "FRANKENSTEIN_PERT"].index[0]
    
    # 1. Feature Vector Space (Absolute Determinism)
    # This must be EXACTLY zero. UMAP min_dist repels identical points, so we test determinism before squashing.
    x_target = X[target_idx]
    x_clone = X[clone_idx]
    clone_feature_dist = np.linalg.norm(x_target - x_clone)
    
    # 2. UMAP Projection Space (Topological Distance)
    t_coords = proj[target_idx]
    c_coords = proj[clone_idx]
    p_coords = proj[pert_idx]
    
    clone_umap_dist = np.linalg.norm(t_coords - c_coords)
    pert_umap_dist = np.linalg.norm(t_coords - p_coords)
    
    print("\n" + "-"*30)
    print("       DIAGNOSTIC RESULTS       ")
    print("-"*30)
    
    table = pd.DataFrame({
        "Condition": ["Identical Twin Clone", "Shuffled Frankenstein"],
        "Extraction_Math_Error": [f"{clone_feature_dist:.6f}", "N/A"],
        "Euclidean_UMAP_Ejection": [f"{clone_umap_dist:.6f} (min_dist repulsion)", f"{pert_umap_dist:.6f}"]
    })
    
    print(table.to_markdown(index=False))
    
    print("\n> INVARIANT CHECKS:")
    print(f"  - Deterministic Math Constraint: Clone vector extraction error 0.0 -> {'PASS' if clone_feature_dist < 0.001 else 'FAIL'}")
    print(f"  - Topological Ejection Constraint: Frankenstein UMAP shifted >1.0 -> {'PASS' if pert_umap_dist > 1.0 else 'FAIL'}")
    print("="*50 + "\n")
    
    # Visually Plot verification
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_meta, x="UMAP_X", y="UMAP_Y", s=100, color="gray", alpha=0.5, label="Normal Population Nodes")
    plt.scatter(t_coords[0], t_coords[1], color="#3498db", s=250, marker="*", label="True Source Node")
    plt.scatter(c_coords[0], c_coords[1], color="none", edgecolor="#3498db", s=400, linewidth=2, label="Clone Perimeter (Distance 0)")
    plt.scatter(p_coords[0], p_coords[1], color="#e74c3c", s=250, marker="X", label="Frankenstein Ejection")
    
    plt.legend()
    plt.title("UMAP Defense Math: Topological Invariance vs Ejection", fontsize=15, fontweight="bold")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    out_dir = os.path.join(os.path.dirname(__file__), "../output/clustering/amph")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "clustering_verification_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved clustering_verification_diagnostics.png to output/clustering/amph/")

def get_domain_combos():
    combos = []
    # 1 to 7 corresponding to binary 001 to 111
    # Bits: 1=Cov, 2=Strat, 4=Temporal
    for i in range(1, 8):
        name_parts = []
        c_cov  = bool(i & 1)
        c_strat = bool(i & 2)
        c_temp = bool(i & 4)
        
        if c_cov: name_parts.append("Cov")
        if c_strat: name_parts.append("Strat")
        if c_temp: name_parts.append("Temp")
        
        c_name = f"{i:02d}_" + "_".join(name_parts)
        combos.append({
            "id": i,
            "name": c_name,
            "use_cov": c_cov,
            "use_strat": c_strat,
            "use_temp": c_temp
        })
    return combos

def silhouette_permutation_test(coords, labels, n_permutations=2000):
    from sklearn.metrics import silhouette_score
    
    # Needs at least 2 classes
    if len(np.unique(labels)) < 2:
        return 0.0, np.zeros(n_permutations), 1.0
        
    true_score = silhouette_score(coords, labels)
    
    null_scores = np.zeros(n_permutations)
    labels_copy = labels.copy()
    
    for i in range(n_permutations):
        np.random.shuffle(labels_copy)
        null_scores[i] = silhouette_score(coords, labels_copy)
        
    p_value = np.mean(null_scores >= true_score)
    p_value = max(p_value, 1/n_permutations)
    return true_score, null_scores, p_value

def _process_combo(combo, df_feat_master, df_meta, df_valid, project_name):
    print(f" > Generating UMAP Domain: {combo['name']}")
    
    # Filter feature columns based on combinatorial domains
    cols = [c for c in df_feat_master.columns if c.startswith("phys_")]
    if combo["use_cov"]: cols += [c for c in df_feat_master.columns if c.startswith("cov_")]
    if combo["use_strat"]: cols += [c for c in df_feat_master.columns if c.startswith("strat_")]
    if combo["use_temp"]: cols += [c for c in df_feat_master.columns if c.startswith("temp_")]
    
    df_feat = df_feat_master[cols].iloc[df_valid]
    
    # Standardize
    X = StandardScaler().fit_transform(df_feat.fillna(0))
    
    # Non-Linear Dimension Squashing
    n_nb = min(15, len(df_valid) - 1)
    reducer = umap.UMAP(n_neighbors=n_nb, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(X)
    
    # We assign projection locally so we don't mess up other jobs
    df_plot = df_meta.iloc[df_valid].copy()
    df_plot["UMAP_X"] = proj[:, 0]
    df_plot["UMAP_Y"] = proj[:, 1]
    
    # Generate completely independent topological spaces (scatterplots) for each Drug state
    drugs = df_plot["Drug"].unique().tolist()
    for drug in drugs:
        df_drug = df_plot[df_plot["Drug"] == drug].copy()
        if df_drug.empty: continue
        
        plt.figure(figsize=(10, 8))
        
        # Color palettes
        genos = sorted(df_drug["Genotype"].dropna().unique().tolist())
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f1c40f"]
        palette = {g: colors[idx % len(colors)] for idx, g in enumerate(genos)}
        
        sns.scatterplot(
            data=df_drug, x="UMAP_X", y="UMAP_Y", 
            hue="Genotype", s=250, alpha=0.85, edgecolor="black", linewidth=1.2,
            palette=palette
        )
        
        # Superimpose Animal ID
        for _, row in df_drug.iterrows():
            plt.text(row["UMAP_X"], row["UMAP_Y"] + 0.1, str(row["Animal"]), 
                     fontsize=9, alpha=0.8, ha="center", va="bottom", zorder=10)
        
        # Calculate statistical significance for legend
        if len(genos) >= 2:
            coords = df_drug[["UMAP_X", "UMAP_Y"]].values
            labels = df_drug["Genotype"].values
            true_score, null_scores, p_val = silhouette_permutation_test(coords, labels, n_permutations=2000)
            
            # Place stats in lower left as text
            p_text = f"p < 0.001" if p_val <= 0.0005 else f"p = {p_val:.4f}"
            stat_str = f"Silhouette: {true_score:.3f}\nEmpirical {p_text}"
            
            plt.gca().text(0.05, 0.05, stat_str, transform=plt.gca().transAxes,
                           fontsize=10, fontweight='bold', verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.title(f"Behavioral Domain: {combo['name']} ({drug})", fontsize=15, fontweight="bold")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()
        
        # Save into isolated subdirectory
        proj = project_name.lower().split("_")[0]
        out_dir = os.path.join(os.path.dirname(__file__), f"../output/clustering/{proj}/{combo['name']}")
        os.makedirs(out_dir, exist_ok=True)
        drug_clean = drug.lower().replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_umap_{drug_clean}_cluster.png"), dpi=300)
        plt.close()
        
        # Save verbatim statistics to text file
        stats_path = os.path.join(out_dir, "cluster_stats.txt")
        with open(stats_path, "a") as f:
            f.write(f"--- Project: {project_name} | Drug: {drug} ---\n")
            if len(genos) >= 2:
                f.write(f"True Silhouette Score: {true_score:.6f}\n")
                f.write(f"Empirical P-Value (N=2000): {p_val:.6f}\n")
            else:
                f.write("Insufficient genotypes (>= 2) for silhouette test.\n")
            f.write("\n")

def run_discovery(sessions, project_name="AMPH"):
    from joblib import Parallel, delayed
    print(f"\n--- Running UMAP Combinatorial Genotype Discovery ---")
    
    # 1. Identify highest occurring motifs globally across dataset
    mapping = motifs.get_label_mapping(sessions, style="standard")
    trigrams_all = pd.Series(dtype=float)
    for s in sessions:
        t = motifs.extract_ngrams(s, n=3, mapping=mapping)
        if not t.empty: trigrams_all = trigrams_all.add(t, fill_value=0)
        
    # We trace 30 structural motifs
    top_30 = trigrams_all.sort_values(ascending=False).head(30).index.tolist()
    
    # 2. Extract massive vectors for all sessions
    df_feat_master, df_meta = build_feature_matrix(sessions, top_30, mapping=mapping)
    
    valid_genos = [g for g in df_meta["Genotype"].unique() if g not in ["UNK", "None", None, "nan"]]
    df_valid = df_meta[df_meta["Genotype"].isin(valid_genos)].index
    if len(df_valid) == 0:
        print(f"No valid genotypes found in {df_meta['Genotype'].unique()}. Bailing!")
        return
        
    combos = get_domain_combos()
    
    Parallel(n_jobs=-1)(
        delayed(_process_combo)(combo, df_feat_master, df_meta, df_valid, project_name) 
        for combo in combos
    )
            
    print(f"Saved Genotype spatial projection clustering plots across {len(combos)} domains in parallel!")
