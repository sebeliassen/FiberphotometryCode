import numpy as np
from scipy.optimize import curve_fit
from scipy.special import logsumexp

import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight
from fiberphotometry.config import LETTER_TO_FREQS, PLOTTING_CONFIG, SYNC
import fiberphotometry.config as config

class PlottingSetup:
    def __init__(self, baseline_duration, trial_length, fps, fit_window_start, fit_window_end):
        # just store, assume caller validated
        self.baseline_duration_in_mins = baseline_duration
        self.trial_length_in_mins = trial_length
        self.photometry_fps = fps
        self.fit_window_start = fit_window_start
        self.fit_window_end = fit_window_end

    def setup_plotting_attributes(self, session, freq):
        ref_freq = config.SYNC.get("reference_phot_freq", freq)
        phot_df = session.dfs.get_data(f"phot_{ref_freq}")
        if phot_df is None or phot_df.empty:
            phot_df = session.dfs.get_data(f"phot_{freq}")

        # compute trial_start_idx
        if session.session_type == 'cpt':
            raw_df = session.dfs.get_data("raw")
            sec_zero_col = config.SYNC['sec_zero_col']
            sync_time_aligned = raw_df.loc[session.cpt, sec_zero_col]
            zero_times = phot_df[sec_zero_col].values
            session.trial_start_idx = int(np.searchsorted(zero_times, sync_time_aligned))

        elif session.session_type == 'oft':
            cam_col = 'cam_frame_num'
            indices = np.where(phot_df[cam_col] >= session.inj_start_time)[0]
            session.trial_start_idx = indices[0]

        else:
            # let it error if invalid session_type
            session.trial_start_idx = None

        # compute frame counts
        coeff = int(self.photometry_fps * 60)
        baseline_frames = coeff * self.baseline_duration_in_mins
        trial_frames   = coeff * self.trial_length_in_mins
        fit_start_off  = coeff * self.fit_window_start
        fit_end_off    = coeff * self.fit_window_end

        # raw indices
        tsi = session.trial_start_idx
        session.plot_start_full = tsi - baseline_frames
        session.plot_end_full   = tsi + trial_frames
        session.fit_start       = tsi - fit_start_off
        session.fit_end         = tsi - fit_end_off
        session.fitting_interval = [session.fit_start, session.fit_end]

    def apply_phot_iso_calculation(self, session, func, phot_df, iso_df):
        fit_range  = range(session.fit_start, session.fit_end)
        plot_range = range(session.plot_start_full, session.plot_end_full)
        for brain_region in session.brain_regions:
            if brain_region in phot_df.columns:
                func(phot_df, iso_df, brain_region, fit_range, plot_range)

    def calculate_dff_continous_iso(self, phot_df, iso_df, brain_region, fit_range, plot_range):
        phot_signal = phot_df[brain_region]
        iso_signal  = iso_df[brain_region]

        fit_idx  = list(fit_range)
        plot_idx = list(plot_range)

        # mean correction
        mean_phot_fit = phot_signal.iloc[fit_idx].mean()
        mean_iso_fit  = iso_signal.iloc[fit_idx].mean()
        mean_diff     = mean_phot_fit - mean_iso_fit

        # phot minus iso
        region_corr = phot_signal - iso_signal + mean_diff

        # shift so min is zero
        min_pos = region_corr.iloc[plot_idx].min()
        region_positive = region_corr + abs(min_pos)

        # ΔF/F
        baseline = region_positive.iloc[fit_idx].mean()
        dff = (region_positive - baseline) / baseline

        # z-score
        z_base = dff.iloc[fit_idx].mean()
        z_std  = dff.iloc[fit_idx].std(ddof=1)
        phot_df[brain_region + ('phot_zF',)] = (dff - z_base) / z_std

    def calculate_dff_and_zscore(self, phot_df, iso_df, brain_region, fit_range, plot_range):
        raw_phot = phot_df[brain_region]
        raw_iso  = iso_df[brain_region]

        baseline_diff = raw_phot[fit_range].mean() - raw_iso[fit_range].mean()
        corrected     = raw_phot - raw_iso + baseline_diff

        min_in_window = corrected[plot_range].min()
        offset = min_in_window if min_in_window < 0 else 0
        positive_only = corrected - offset

        baseline_raw = positive_only[fit_range].mean()
        dff = (positive_only - baseline_raw) / baseline_raw

        z_baseline = dff[fit_range].mean()
        z_std      = dff[fit_range].std(ddof=1)
        phot_df[brain_region + ('phot_zF',)] = (dff - z_baseline) / z_std

    def calculate_dff_exp2_iso(self, phot_df, iso_df, brain_region, fit_range, plot_range):
        phot_signal   = phot_df[brain_region].iloc[plot_range]
        iso_signal    = iso_df[brain_region].iloc[plot_range]
        x_all         = np.arange(len(iso_signal))

        def double_exponential(x, a1, b1, a2, b2):
            max_exp = 700
            log1 = np.clip(np.log(a1) + b1 * x, -max_exp, max_exp)
            log2 = np.clip(np.log(a2) + b2 * x, -max_exp, max_exp)
            return np.exp(logsumexp([log1, log2], axis=0))

        valid = np.isfinite(iso_signal)
        x_valid = x_all[valid]
        y_valid = iso_signal[valid]
        params, _ = curve_fit(
            double_exponential, x_valid, y_valid,
            p0=[1, -0.1, 1, -0.1],
            bounds=([0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
        )

        iso_fit = double_exponential(x_all, *params)
        X = sm.add_constant(iso_fit)
        rlm_results = sm.RLM(phot_signal, X, M=TukeyBiweight()).fit()
        intercept, slope = rlm_results.params

        lin_fit = slope * iso_fit + intercept
        dFF_signal = (phot_signal - lin_fit) / lin_fit
        zscore_dff = (dFF_signal - dFF_signal.mean()) / dFF_signal.std()

        phot_df[brain_region + ('phot_zF',)] = zscore_dff

    def apply_plotting_setup_to_sessions(self, sessions):
        for session in sessions:
            for letter, freq in LETTER_TO_FREQS.items():
                if letter == 'iso':
                    continue
                self.setup_plotting_attributes(session, freq)
                phot_df = session.dfs.get_data(f"phot_{freq}")
                iso_df  = session.dfs.get_data(f"phot_{LETTER_TO_FREQS['iso']}")
                self.apply_phot_iso_calculation(session, self.calculate_dff_continous_iso, phot_df, iso_df)