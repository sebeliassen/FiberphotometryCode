from scipy.stats import ttest_ind
from tqdm.notebook import tqdm
import numpy as np
from processing.session_sampling import MiceAnalysis


def sessions_t_test(sessions):
    mouse_analyser = MiceAnalysis(sessions)

    results = []
    n = 1000

    for metric in ('c_score', 'd_prime', 'participation'):     
        for brain_region in ['VS', 'DMS', 'DLS']:
            for event_type in ['hit', 'mistake', 'miss', 'cor_reject', 'reward_collect']:
                lo_responses, hi_responses = mouse_analyser.sample_response_metrics(metric, brain_region, event_type, n=n)
                # Perform t-test
                for response_metric in lo_responses.keys():
                    curr_lo_responses = lo_responses[response_metric]
                    curr_hi_responses = hi_responses[response_metric]
                    t_statistic, p_value = ttest_ind(curr_lo_responses, 
                                                    curr_hi_responses, nan_policy='omit')
                    result = {
                        'key': (metric, brain_region, event_type, response_metric),
                        'T-Statistic': t_statistic,
                        'P-Value': p_value,
                        'n': min(n, len(curr_lo_responses), len(curr_hi_responses)),
                        'low_high_vals': (sum(curr_lo_responses) / len(curr_lo_responses), 
                                          sum(curr_hi_responses) / len(curr_hi_responses))
                    }
                    if np.isnan(p_value):
                        continue
                    results.append(result)

    sorted_results = sorted(results, key=lambda x: x['P-Value'])
    return sorted_results