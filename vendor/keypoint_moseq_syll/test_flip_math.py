keys = [
    '20240728_100311_1664_1',
    '20240728_100311_1665_1',
    '20240740_something', # Just an example
]
# Let's get actual keys
import glob

results_path = glob.glob('/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project/*/results.h5')
if results_path:
    import keypoint_moseq as kpms
    results = kpms.load_results('/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project', 'optimized_flip_activity_k100k')
    keys = list(results.keys())

    OPTIMAL_FLIP_IDS = [
        728,
        740,
        1767,
        1773,
        1664,
        1666,
        1779,
        1781
    ]
    flip_set = set(str(x) for x in OPTIMAL_FLIP_IDS)

    n_flipped = 0
    for k in keys:
        should_flip = False
        matching_fids = []
        for fid in flip_set:
            if fid in k:
                should_flip = True
                matching_fids.append(fid)

        if should_flip:
            n_flipped += 1
            print(f"[{k}] flipped due to matching fids: {matching_fids}")
        else:
            print(f"[{k}] NOT FLIPPED")

    print(f"\nTotal Flipped: {n_flipped}/{len(keys)}")
