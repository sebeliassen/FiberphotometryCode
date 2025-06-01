import warnings
import numpy as np
import re
from scipy.optimize import curve_fit
from scipy.special import logsumexp

import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight
from fiberphotometry.config import LETTER_TO_FREQS
import fiberphotometry.config as config

class PlottingSetup:
    def __init__(self, baseline_duration, trial_length, fps, fit_window_start, fit_window_end):
        # --- generic non-negativity checks ---
        non_negative = {
            'baseline_duration': baseline_duration,
            'trial_length':     trial_length,
            'fit_window_start': fit_window_start,
            'fit_window_end':   fit_window_end,
        }
        for name, val in non_negative.items():
            if val < 0:
                raise ValueError(f"{name!r} must be ≥ 0 (got {val})")

        # --- fps must be strictly positive ---
        if fps <= 0:
            raise ValueError(f"fps must be positive (got {fps})")

        # --- relational check ---
        if fit_window_start <= fit_window_end:
            raise ValueError(
                f"fit_window_start ({fit_window_start}) must be < fit_window_end ({fit_window_end})"
            )

        # store after validation
        self.baseline_duration_in_mins = baseline_duration
        self.trial_length_in_mins      = trial_length
        self.photometry_fps            = fps
        self.fit_window_start          = fit_window_start
        self.fit_window_end            = fit_window_end

    def setup_plotting_attributes(self, session, freq):
        if session.session_type != 'cpt':
            raise NotImplementedError(
                f"session_type='{session.session_type}' not supported—only 'cpt' is implemented"
            )

        ref_freq = config.SYNC.get("reference_phot_freq", freq)
        phot_df = session.dfs.get_data(f"phot_{ref_freq}")
        if phot_df is None:
            warnings.warn(
                f"No photometry DataFrame found for reference freq '{ref_freq}'; "
                f"falling back to phot_{freq}"
            )
            phot_df = session.dfs.get_data(f"phot_{freq}")
        elif phot_df.empty:
            warnings.warn(
                f"Photometry DataFrame for reference freq '{ref_freq}' is empty; "
                f"falling back to phot_{freq}"
            )
            phot_df = session.dfs.get_data(f"phot_{freq}")

        # compute trial_start_idx
        if session.session_type == 'cpt':
            raw_df = session.dfs.get_data("raw")
            if raw_df is None:
                raise ValueError("No raw DataFrame available for CPT session")

            sec_zero_col = config.SYNC.get('sec_zero_col')
            if sec_zero_col not in raw_df.columns:
                raise KeyError(f"Column '{sec_zero_col}' not found in raw DataFrame")
            if sec_zero_col not in phot_df.columns:
                raise KeyError(f"Column '{sec_zero_col}' not found in photometry DataFrame for '{ref_freq}'")

            zero_times = phot_df[sec_zero_col].values
            sync_time_aligned = raw_df.loc[session.cpt, sec_zero_col]
            idx = np.searchsorted(zero_times, sync_time_aligned)
            if idx >= len(zero_times):
                raise ValueError(
                    f"sync_time {sync_time_aligned} (row {session.cpt}) is beyond photometry timestamps; "
                    f"insertion idx={idx} but max index is {len(zero_times)-1}"
                )
            session.trial_start_idx = int(idx)

        elif session.session_type == 'oft':
            cam_col = 'cam_frame_num'
            indices = np.where(phot_df[cam_col] >= session.inj_start_time)[0]
            session.trial_start_idx = indices[0]

        else:
            session.trial_start_idx = None

        # compute frame counts
        coeff = int(self.photometry_fps * 60)
        baseline_frames = int(coeff * self.baseline_duration_in_mins)
        trial_frames   = int(coeff * self.trial_length_in_mins)
        fit_start_off  = int(coeff * self.fit_window_start)
        fit_end_off    = int(coeff * self.fit_window_end)

        # raw indices
        tsi = session.trial_start_idx
        session.plot_start_full = tsi - baseline_frames
        session.plot_end_full   = tsi + trial_frames
        session.fit_start       = tsi - fit_start_off
        session.fit_end         = tsi - fit_end_off
        session.fitting_interval = [session.fit_start, session.fit_end]

        n = len(phot_df)
        if session.plot_start_full < 0:
            max_baseline_mins = tsi / coeff
            raise IndexError(
                f"baseline_duration_in_mins={self.baseline_duration_in_mins} → "
                f"{baseline_frames} frames exceeds trial_start_idx={tsi} frames "
                f"({tsi/coeff:.2f} mins). "
                f"Set baseline_duration ≤ {max_baseline_mins:.2f} minutes."
            )
        if session.plot_end_full > n:
            max_trial_mins = (n - tsi) / coeff
            raise IndexError(
                f"trial_length_in_mins={self.trial_length_in_mins} → "
                f"{trial_frames} frames extends beyond data length={n} frames "
                f"({n/coeff:.2f} mins). Set trial_length ≤ {max_trial_mins:.2f} minutes."
            )

        for name, start, end in [
            ("full plot window start", session.plot_start_full, None),
            ("full plot window end",   None, session.plot_end_full),
            ("fit window start",       session.fit_start, None),
            ("fit window end",         None, session.fit_end),
        ]:
            if start is not None and (start < 0 or start >= n):
                raise IndexError(f"{name} index {start} out of bounds for DataFrame length {n}")
            if end   is not None and (end   < 0 or end   >  n):
                raise IndexError(f"{name} index {end} out of bounds for DataFrame length {n}")

    def apply_phot_iso_calculation(self, session, func, phot_df, iso_df):
        """
        Call `func` (e.g., calculate_dff_continous_iso) on every column
        in phot_df whose name matches the regex r"^signal_\\d+$".
        """
        fit_range  = range(session.fit_start, session.fit_end)
        plot_range = range(session.plot_start_full, session.plot_end_full)

        pattern = re.compile(r"^signal_\d+$")
        for col in phot_df.columns:
            if pattern.match(col):
                func(phot_df, iso_df, col, fit_range, plot_range)

    def calculate_dff_continous_iso(self, phot_df, iso_df, signal_col, fit_range, plot_range):
        # Extract the two raw streams
        phot_signal = phot_df[signal_col]
        iso_signal  = iso_df[signal_col]

        fit_idx  = list(fit_range)
        plot_idx = list(plot_range)

        # Mean correction: subtract iso from phot and add back mean diff
        mean_phot_fit = phot_signal.iloc[fit_idx].mean()
        mean_iso_fit  = iso_signal.iloc[fit_idx].mean()
        mean_diff     = mean_phot_fit - mean_iso_fit

        region_corr = phot_signal - iso_signal + mean_diff

        # Shift so minimum in plot window is zero
        min_pos = region_corr.iloc[plot_idx].min()
        region_positive = region_corr + abs(min_pos)

        # ΔF/F
        baseline = region_positive.iloc[fit_idx].mean()
        dff = (region_positive - baseline) / baseline

        # z-score within fit window
        z_base = dff.iloc[fit_idx].mean()
        z_std  = dff.iloc[fit_idx].std(ddof=1)

        phot_df[f"{signal_col}_dff"] = (dff - z_base) / z_std

    def calculate_dff_and_zscore(self, phot_df, iso_df, signal_col, fit_range, plot_range):
        raw_phot = phot_df[signal_col]
        raw_iso  = iso_df[signal_col]

        # ——————————————
        # 0) Fill any NaNs in the input streams by linear interpolation
        #     (limit_direction='both' ensures edge NaNs are also filled).
        raw_phot = raw_phot.interpolate(method='linear', limit_direction='both')
        raw_iso  = raw_iso.interpolate(method='linear', limit_direction='both')

        # 1) Compute baseline_diff, guarding against an empty fit window
        phot_fit = raw_phot.iloc[fit_range]
        iso_fit  = raw_iso.iloc[fit_range]

        # If fit window is empty (no indices) or still all NaN, fill output with zeros
        if phot_fit.size == 0 or iso_fit.size == 0 or phot_fit.isna().all() or iso_fit.isna().all():
            zeros = np.zeros(len(raw_phot))
            phot_df[f"{signal_col}_dff"] = zeros
            return

        mean_phot_fit = phot_fit.mean()
        mean_iso_fit  = iso_fit.mean()
        baseline_diff = mean_phot_fit - mean_iso_fit

        # 2) Corrected trace
        corrected = raw_phot - raw_iso + baseline_diff

        # 3) Compute offset = min(corrected in plot window) if negative
        corr_plot = corrected.iloc[plot_range]
        if corr_plot.size == 0:
            offset = 0.0
        else:
            min_val = corr_plot.min(skipna=True)
            if not np.isfinite(min_val):
                offset = 0.0
            else:
                offset = min_val if min_val < 0 else 0.0

        positive_only = corrected - offset

        # 4) Compute baseline_raw from the fit window
        po_fit = positive_only.iloc[fit_range]
        if po_fit.size == 0 or po_fit.isna().all():
            baseline_raw = 0.0
        else:
            baseline_raw = po_fit.mean()

        # 5) Guard against division-by-zero or non-finite baseline_raw
        if not np.isfinite(baseline_raw) or baseline_raw == 0:
            zeros = np.zeros(len(positive_only))
            phot_df[f"{signal_col}_dff"] = zeros
            return

        # 6) Compute ΔF/F
        dff = (positive_only - baseline_raw) / baseline_raw

        # 7) Compute z-score over the fit window
        dff_fit = dff.iloc[fit_range]
        if dff_fit.size == 0 or dff_fit.isna().all():
            zeros = np.zeros(len(dff))
            phot_df[f"{signal_col}_dff"] = zeros
            return

        z_baseline = dff_fit.mean()
        z_std      = dff_fit.std(ddof=1)

        # 8) Guard against zero or NaN standard deviation
        if not np.isfinite(z_std) or z_std == 0:
            zeros = np.zeros(len(dff))
            phot_df[f"{signal_col}_dff"] = zeros
            return

        # 9) Save z-scored ΔF/F
        phot_df[f"{signal_col}_dff"] = (dff - z_baseline) / z_std



    def calculate_dff_exp2_iso(self, phot_df, iso_df, signal_col, fit_range, plot_range):
        phot_signal = phot_df[signal_col].iloc[plot_range]
        iso_signal  = iso_df[signal_col].iloc[plot_range]
        x_all        = np.arange(len(iso_signal))

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

        phot_df[f"{signal_col}_dff"] = zscore_dff

    def apply_plotting_setup_to_sessions(self, sessions):
        for session in sessions:
            for letter, freq in LETTER_TO_FREQS.items():
                if letter == 'iso':
                    continue

                # Compute trial start indexes and plotting/fitting windows
                self.setup_plotting_attributes(session, freq)

                phot_df = session.dfs.get_data(f"phot_{freq}")
                iso_df  = session.dfs.get_data(f"phot_{LETTER_TO_FREQS['iso']}")

                # For every column matching "signal_<number>", compute ΔF/F + z-score
                self.apply_phot_iso_calculation(
                    session,
                    self.calculate_dff_and_zscore,
                    phot_df,
                    iso_df
                )
