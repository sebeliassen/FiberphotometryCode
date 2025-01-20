import numpy as np
from scipy.optimize import curve_fit
from scipy.special import logsumexp

import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight
from config import LETTER_TO_FREQS

class PlottingSetup:
    def __init__(self, baseline_duration, trial_length, fps, fit_window_start, fit_window_end):
        self.baseline_duration_in_mins = baseline_duration
        self.trial_length_in_mins = trial_length
        self.photometry_fps = fps

        self.fit_window_start = fit_window_start
        self.fit_window_end = fit_window_end


    def setup_plotting_attributes(self, session, freq):
        phot_df = session.dfs.get_data(f"phot_{freq}")
        if session.session_type == 'cpt':
            session.trial_start_sync = phot_df["SecFromZero_FP3002"] - session.set_blank_images_timepoint_fp3002
            session.trial_start_idx = np.argmax(session.trial_start_sync > 0)
        elif session.session_type == 'oft':
            session.trial_start_idx = np.argmax(phot_df['cam_frame_num'] >= session.inj_start_time)
        
        # Calculate the start and end points for the full plot
        mins_to_frames_coeff = self.photometry_fps * 60
        session.plot_start_full = session.trial_start_idx - mins_to_frames_coeff * self.baseline_duration_in_mins
        session.plot_end_full = session.trial_start_idx + mins_to_frames_coeff * self.trial_length_in_mins

        session.fitting_interval = [session.trial_start_idx - self.fit_window_start * mins_to_frames_coeff,
                                    session.trial_start_idx - self.fit_window_end * mins_to_frames_coeff]

        session.fit_start, session.fit_end = session.fitting_interval
        
    def apply_phot_iso_calculation(self, session, func, phot, iso):
        for brain_region in session.brain_regions:
            if brain_region not in phot.columns:
                continue
            func(phot, iso, brain_region,
                 range(session.fit_start, session.fit_end), 
                 range(session.plot_start_full, session.plot_end_full))

    def calculate_dff_continous_iso(self, phot_df, iso_df, brain_region, fit_range, plot_range):
        phot_signal = phot_df[brain_region]
        iso_signal = iso_df[brain_region]

        mean_diff = (phot_signal[fit_range].mean() - iso_signal[fit_range].mean())

        # Apply the mean difference to the entire phot brain region column to adjust it against the iso region
        region_phot_minus_iso = phot_signal - iso_signal + mean_diff

        # Adjust delta F/F to only include positive values
        min_positive_dFF = abs(region_phot_minus_iso[plot_range].min())
        region_phot_dF_onlypositive = region_phot_minus_iso + min_positive_dFF

        # Calculate the z-scored signal for the phot brain region
        mean_dF_onlypositive = region_phot_dF_onlypositive[fit_range].mean()
        std_dF_onlypositive = region_phot_dF_onlypositive[fit_range].std()
        region_phot_zF = (region_phot_dF_onlypositive - mean_dF_onlypositive) / std_dF_onlypositive
        phot_df[brain_region + ('phot_zF',)] = region_phot_zF

    def calculate_dff_exp2_iso(self, phot_df, iso_df, brain_region, fit_range, plot_range):
        phot_signal = phot_df[brain_region]
        phot_trimmed = phot_signal[plot_range]

        iso_signal = iso_df[brain_region]
        iso_trimmed = iso_signal[plot_range]

        def double_exponential(x, a1, b1, a2, b2):
            # Clamp large values to prevent overflow
            max_exp = 700
            log_term1 = np.clip(np.log(a1) + b1 * x, -max_exp, max_exp)
            log_term2 = np.clip(np.log(a2) + b2 * x, -max_exp, max_exp)
            return np.exp(logsumexp([log_term1, log_term2], axis=0))

        # Create x-data corresponding to iso_trimmed indices.
        x_all = np.arange(len(iso_trimmed))
        
        # Identify valid points (exclude NaN/infinite values)
        valid = np.isfinite(iso_trimmed)
        if not np.any(valid):
            raise RuntimeError(f"No valid data points for curve fitting in brain region {brain_region}")
        if np.sum(valid) < 10:
            raise RuntimeError(f"Not enough valid data points ({np.sum(valid)}) for curve fitting in brain region {brain_region}")

        x_valid = x_all[valid]
        y_valid = iso_trimmed[valid]

        initial_guess = [1, -0.1, 1, -0.1]

        try:
            bounds = ([0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
            params, _ = curve_fit(double_exponential, x_valid, y_valid, p0=initial_guess, bounds=bounds)
        except Exception as e:
            raise RuntimeError(f"curve_fit failed for {brain_region}: {e}")

        # Evaluate iso_fit on the full domain
        iso_fit = double_exponential(x_all, *params)
        if not np.all(np.isfinite(iso_fit)):
            raise RuntimeError(f"Non-finite values encountered in iso_fit for {brain_region} after fitting.")

        # Fit a robust linear model to relate phot_trimmed ~ iso_fit.
        X = sm.add_constant(iso_fit)
        try:
            rlm_model = sm.RLM(phot_trimmed, X, M=TukeyBiweight())
            rlm_results = rlm_model.fit()
        except Exception as e:
            raise RuntimeError(f"RLM failed for {brain_region}: {e}")

        if len(rlm_results.params) != 2:
            raise RuntimeError(f"Unexpected RLM parameter count for {brain_region}: {rlm_results.params}")

        intercept, slope = rlm_results.params
        lin_fit = slope * iso_fit + intercept

        # Compute dF/F (here, normalized against the iso_fit)
        baseline_mean = np.mean(phot_signal[fit_range])
        try:
            dFF_signal = (phot_trimmed - lin_fit) / lin_fit
        except Exception as e:
            raise RuntimeError(f"dFF calculation failed for {brain_region}: {e}")

        dFF_mean = dFF_signal.mean()
        dFF_std = dFF_signal.std()
        region_phot_zF = (dFF_signal - dFF_mean) / dFF_std
        phot_df[brain_region + ('phot_zF',)] = region_phot_zF
    
    def apply_plotting_setup_to_sessions(self, sessions):
        for session in sessions:
            for letter, freq in LETTER_TO_FREQS.items():
                if letter == 'iso':
                    continue
                self.setup_plotting_attributes(session, freq)
                phot_df = session.dfs.get_data(f"phot_{freq}")
                iso_df = session.dfs.get_data(f"phot_{LETTER_TO_FREQS['iso']}")
                self.apply_phot_iso_calculation(session, self.calculate_dff_exp2_iso, phot_df, iso_df)
                # self.apply_phot_iso_calculation(session, self.calculate_dff_continous_iso, phot_df, iso_df)