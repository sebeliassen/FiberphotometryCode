import re

class Renamer:
    @staticmethod
    def rename_df_columns(df, pattern, replacement):
        df.rename(columns=lambda x: re.sub(pattern, replacement, x), inplace=True)

    @staticmethod
    def rename_sessions_data(sessions, patterns):
        for session in sessions:
            df_keys = session.df_container.fetch_all_data_names()
            for df_type, pattern_info in patterns:
                target_df_keys = [key for key in df_keys if df_type in key]
                for df_key in target_df_keys:            
                    df = session.df_container.get_data(df_key)
                    Renamer.rename_df_columns(df, 
                                              pattern_info.get("pattern", ""), 
                                              pattern_info.get("replacement", ""))
    
    @staticmethod
    def extract_region_number(column_name, letter):
        if letter == 'iso':
            letter = ''
        match = re.search(r'Region(\d+)' + letter, column_name)
        return match.group(1) if match else None

    @staticmethod
    # def rename_sessions_fiber_to_brain_region(sessions, frequencies):
    #     for session in sessions:
    #         for letter, freq in frequencies.items():
    #             photwrit_df_key = f'photwrit_{freq}'
    #             bonsai_df_key = f'bonsai_{freq}'
    #             for df_key in [photwrit_df_key, bonsai_df_key]:
    #                 df = session.df_container.get_data(df_key)
    #                 df.rename(columns=lambda x: 
    #                           session.fiber_to_region.get(Renamer.extract_region_number(x, letter), x),
    #                           inplace=True)

    # To accomodate the openfield-cpt merge, we have changed the functino below to only use the photwrit df
    def rename_sessions_fiber_to_brain_region(sessions, frequencies):
        for session in sessions:
            for letter, freq in frequencies.items():
                phot_df_key = f'phot_{freq}'
                df = session.dfs.get_data(phot_df_key)
                if df is None:
                    continue
                df.rename(columns=lambda x: 
                            session.fiber_to_region.get(Renamer.extract_region_number(x, letter), x),
                            inplace=True)
                    
    @staticmethod
    def debug_df_renames(session):
        df_names = session.df_container.fetch_all_data_names()
        print(f"Debugging session: {session.trial_id}")
        for df_name in df_names:
            df = session.df_container.get_data(df_name)
            if df is not None:
                print(f"Columns in {df_name}: {list(df.columns)}")
            else:
                print(f"{df_name} is None or does not exist.")


class Syncer:
    @staticmethod
    def calculate_cpt(raw_df):
        for index, row in raw_df.iterrows():
            if row["Item_Name"] == "Set Blank Images":
                return index
        return -1

    @staticmethod
    def sync_session(session):
        raw_df = session.df_container.get_data('raw')
        session.cpt = Syncer.calculate_cpt(raw_df)
        session.sync_time = raw_df.iloc[session.cpt]['Evnt_Time']

        Syncer.sync_dataframes(session)

    @staticmethod
    def sync_dataframes(session):
        ttl_df = session.df_container.get_data('ttl')
        bonsai_470_df = session.df_container.get_data('bonsai_470')
        raw_df = session.df_container.get_data('raw')
        if len(ttl_df) == 0:
            raise IndexError('No sync data loaded, data might be missing')

        ttl_df['SecFromZero_Bonsai'] = ttl_df['Timestamp_Bonsai'] - bonsai_470_df['Timestamp_Bonsai'].iloc[0]
        ttl_df['SecFromZero_FP3002'] = ttl_df['Seconds_FP3002'] - bonsai_470_df['Timestamp_FP3002'].iloc[0]

        session.set_blank_images_timepoint_bonsai = ttl_df['SecFromZero_Bonsai'].iloc[0] - session.sync_time
        session.set_blank_images_timepoint_fp3002 = ttl_df['SecFromZero_FP3002'].iloc[0] - session.sync_time
        raw_df['SecFromZero_Bonsai'] = raw_df['Evnt_Time'] + session.set_blank_images_timepoint_bonsai
        raw_df['SecFromZero_FP3002'] = raw_df['Evnt_Time'] + session.set_blank_images_timepoint_fp3002

        bonsai_470_df['SecFromZero'] = bonsai_470_df['Timestamp_Bonsai'] - bonsai_470_df['Timestamp_Bonsai'].iloc[0]
        
        bonsai_415_df = session.df_container.get_data('bonsai_415')
        bonsai_415_df['SecFromZero'] = bonsai_415_df['Timestamp_Bonsai'] - bonsai_470_df['Timestamp_Bonsai'].iloc[0]

        bonsai_560_df = session.df_container.get_data('bonsai_560')
        bonsai_560_df['SecFromZero'] = bonsai_560_df['Timestamp_Bonsai'] - bonsai_470_df['Timestamp_Bonsai'].iloc[0]

        # Initialize min_length with a large number
        min_length = float('inf')

        # Loop through each frequency and compute required fields while finding min_length
        for freq in [470, 415, 560]:
            photwrit_df = session.df_container.get_data(f'photwrit_{freq}')
            bonsai_df = session.df_container.get_data(f'bonsai_{freq}')
            
            photwrit_df['Timestamp_Bonsai'] = bonsai_df['Timestamp_Bonsai']
            photwrit_df['SecFromZero_Bonsai'] = bonsai_df['SecFromZero']
            photwrit_df['SecFromZero_FP3002'] = photwrit_df['Timestamp'] - photwrit_df['Timestamp'].iloc[0]
            photwrit_df['SecFromTrialStart_Bonsai'] = photwrit_df['SecFromZero_Bonsai'] - session.set_blank_images_timepoint_bonsai
            photwrit_df['SecFromTrialStart_FP3002'] = photwrit_df['SecFromZero_FP3002'] - session.set_blank_images_timepoint_fp3002

            # Update min_length based on current DataFrame lengths
            min_length = min(min_length, len(photwrit_df), len(bonsai_df))

        # Truncate all DataFrames to the minimum length found
        for freq in [470, 415, 560]:
            truncated_bonsai_df = session.df_container.get_data(f'bonsai_{freq}').iloc[:min_length]
            truncated_photwrit_df = session.df_container.get_data(f'photwrit_{freq}').iloc[:min_length]
            
            # Reassign the truncated DataFrames directly back to df_container
            session.df_container.data[f'bonsai_{freq}'] = truncated_bonsai_df
            session.df_container.data[f'photwrit_{freq}'] = truncated_photwrit_df


    @staticmethod
    def apply_sync_to_all_sessions(sessions):
        for session in sessions:
            Syncer.sync_session(session)