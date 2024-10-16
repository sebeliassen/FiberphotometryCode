import numpy as np

class PlottingSetup:
    def __init__(self, baseline_duration, trial_length, fps, fit_window_start, fit_window_end):
        self.baseline_duration_in_mins = baseline_duration
        self.trial_length_in_mins = trial_length
        self.photometry_fps = fps

        self.fit_window_start = fit_window_start
        self.fit_window_end = fit_window_end


    def setup_plotting_attributes(self, session):
        photwrit_470_df = session.df_container.get_data("photwrit_470")
        session.trial_start_sync_FP3002 = photwrit_470_df["SecFromZero_FP3002"] - session.set_blank_images_timepoint_fp3002
        session.trial_start_idx = np.argmax(session.trial_start_sync_FP3002 > 0)
    
        # Calculate the start and end points for the full plot
        mins_to_frames_coeff = self.photometry_fps * 60
        session.plot_start_full = session.trial_start_idx - mins_to_frames_coeff * self.baseline_duration_in_mins
        session.plot_end_full = session.trial_start_idx + mins_to_frames_coeff * self.trial_length_in_mins

        session.fitting_interval = [session.trial_start_idx - self.fit_window_start * mins_to_frames_coeff,
                                    session.trial_start_idx - self.fit_window_end * mins_to_frames_coeff]

        session.fit_start = session.trial_start_idx - self.fit_window_start * mins_to_frames_coeff
        session.fit_end = session.trial_start_idx - self.fit_window_end * mins_to_frames_coeff
        
    def apply_phot_iso_calculation(self, session, func, phot, iso):
        for brain_region in session.fiber_to_region.values():
            func(phot, iso, brain_region,
                 range(session.fit_start, session.fit_end), 
                 range(session.plot_start_full, session.plot_end_full))

    def calculate_dff(self, photwrit_df, iso_df, brain_region, fit_range, plot_range):
        phot_signal = photwrit_df[brain_region]
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
        photwrit_df[f'{brain_region}_phot_zF'] = region_phot_zF
        
    def apply_plotting_setup_to_sessions(self, sessions):
        for session in sessions:
            self.setup_plotting_attributes(session)
            phot_df = session.df_container.get_data("photwrit_470")
            iso_df = session.df_container.get_data("photwrit_415")
            self.apply_phot_iso_calculation(session, self.calculate_dff, phot_df, iso_df)