import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from analysis.replications.panel_1z.lib import utils

logger = logging.getLogger("Panel1z.VigorWaveforms")

# --- Configuration ---
DA_WINDOW_SEC = 1.5
VEL_WINDOW = (0.0, 0.5)
MIN_TRIALS = 10
TARGET_COLOR = "G"
IGNORE_GENOTYPES = ["UNK", "nan", None]

def run(sessions, config):
    """
    Run Vigor-Binned Waveforms Analysis.
    """
    logger.info("Gathering data for Vigor Waveforms...")

    # Structure: syll_data[region][geno][syll] = {'traces': [], 'vigors': []}
    syll_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'traces': [], 'vigors': []})))

    for s in sessions:
        meta = s.metadata
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if str(geno) in IGNORE_GENOTYPES: continue

        df = s.all_data_df
        if df.empty: continue

        fs = meta.get('fs', 30.0)
        da_frames = int(DA_WINDOW_SEC * fs)
        v_end = int(VEL_WINDOW[1] * fs)

        onsets = df["onset"].to_numpy()
        onset_idxs = np.flatnonzero(onsets)
        syll_array = df["syllable"].to_numpy()

        if "velocity_px_s" not in df.columns:
            logger.warning(f"Session {s.session_id} missing velocity_px_s. Skipping.")
            continue
        vel_array = df["velocity_px_s"].to_numpy()

        sig_meta = meta.get("signal_meta", {})
        for sig_key, info in sig_meta.items():
            if str(info.get("orig_letter", "UNK")).upper() != TARGET_COLOR: continue

            region = info.get("brain_reg", "UNK")
            if region == "UNK": continue

            # Select Column
            col = None
            if f"{sig_key}_zscore_datta" in df.columns: col = f"{sig_key}_zscore_datta"
            elif f"{sig_key}_zscore" in df.columns: col = f"{sig_key}_zscore"

            if not col: continue
            sig_data = df[col].to_numpy()
            max_idx = len(sig_data)

            for idx in onset_idxs:
                s_da = idx - da_frames
                e_da = idx + da_frames
                e_vel = idx + v_end
                if s_da < 0 or e_da >= max_idx or e_vel >= max_idx: continue

                trace = sig_data[s_da : e_da]
                if np.isnan(trace).all(): continue

                vigor = np.nanmax(vel_array[idx : e_vel])
                syll = syll_array[idx]

                syll_data[region][geno][syll]['traces'].append(trace)
                syll_data[region][geno][syll]['vigors'].append(vigor)

    logger.info("   Aggregation complete.")

    # 2. Binning & Plotting
    for region, geno_dict in sorted(syll_data.items()):

        fig, axes = plt.subplots(1, len(geno_dict), figsize=(5*len(geno_dict), 5), sharey=True, dpi=150)
        if len(geno_dict) == 1: axes = [axes]

        for ax, (geno, syllables) in zip(axes, sorted(geno_dict.items())):

            # Pool all syllables for this regional genotype
            all_points = []
            for syll, data in syllables.items():
                if len(data['traces']) < MIN_TRIALS: continue

                # Use mean per syllable to avoid over-weighting common syllables
                mu_trace = np.nanmean(data['traces'], axis=0)
                mu_vigor = np.nanmean(data['vigors'])
                all_points.append({'trace': mu_trace, 'vigor': mu_vigor})

            if not all_points:
                ax.text(0.5, 0.5, "No Data", ha='center'); continue

            vigors = [p['vigor'] for p in all_points]
            p33 = np.percentile(vigors, 33)
            p66 = np.percentile(vigors, 66)

            bins = {'Low': [], 'Med': [], 'High': []}
            for p in all_points:
                if p['vigor'] <= p33: bins['Low'].append(p['trace'])
                elif p['vigor'] <= p66: bins['Med'].append(p['trace'])
                else: bins['High'].append(p['trace'])

            # Styling
            colors = {'Low': '#bdd7e7', 'Med': '#6baed6', 'High': '#08519c'}
            if geno == "KI": colors = {'Low': '#fcae91', 'Med': '#fb6a4a', 'High': '#a50f15'}

            time_vec = np.linspace(-DA_WINDOW_SEC, DA_WINDOW_SEC, 2*da_frames)

            for label in ['Low', 'Med', 'High']:
                stack = np.array(bins[label])
                if len(stack) > 0:
                    mu = np.nanmean(stack, axis=0)
                    sem = np.nanstd(stack, axis=0) / np.sqrt(len(stack))

                    # Fix length
                    if len(mu) != len(time_vec):
                         mu = np.interp(time_vec, np.linspace(-DA_WINDOW_SEC, DA_WINDOW_SEC, len(mu)), mu)
                         sem = np.interp(time_vec, np.linspace(-DA_WINDOW_SEC, DA_WINDOW_SEC, len(sem)), sem)

                    ax.plot(time_vec, mu, color=colors[label], linewidth=2, label=label)
                    ax.fill_between(time_vec, mu-sem, mu+sem, color=colors[label], alpha=0.2)

            ax.axvline(0, color="black", linestyle="--", alpha=0.3)
            ax.set_title(f"{region} | {geno}\n(N={len(all_points)} Syllables)")
            ax.set_xlabel("Time from Onset (s)")
            ax.legend(title="Avg Vigor", loc='upper right', fontsize='x-small')

        axes[0].set_ylabel("Neural Signal (Z)")
        sns.despine()
        plt.tight_layout()
        utils.save_plot(fig, f"vigor_waveforms_{region.replace('-','_')}", config)
