# Pooled Karen capacity comparison

15 October 2026. User authorised the intermediate-K sweep and subsequent
quantitative verification. Runtime `status.json`/log are authoritative while
the detached job is active. This is an exploratory common-generator analysis,
not a cohort-specific fit or independent validation set.

## Scope

K6/10/14/18/22, lambda.3/3, seeds42/43/44, both L1/100state and existing grouped
L3_unresolved_separate/69state vocabularies:60configurations.12K10models are
reused byte-for-byte from the completed panel;48new fits use its exact frozen
native matrices/onset masks. L3bout steps and80iterations are unchanged.
Lambda.3 performed well for coverage in the initial panel;3provides a stronger
competition contrast. This pair was fixed before the new Kfits.

All99recordings/all21mice/all5waves, includingL-DOPA, contribute to every pooled
candidate. No subsampling or prior held-out split. Group differences/DA are not
selection criteria. The old source/model implementations and input bundle remain
unchanged. New study owners: `reference_capacity.py`, `run_reference_capacity.py`.

## Diagnostics

- Reconstruction error/uncentred reconstruction score and physical union time
  coverage, with individual seed traces against K. Reconstruction curves remain
  separate by vocabulary; they encode different targets.
- Per-family accepted and activation-candidate template-error p10/median/p90,
  acceptance fraction, execution count, mice and effective number of mice. Candidate
  errors before the cutoff expose truncation in the accepted-error distribution.
- Supported-family count at20calls/5mice, alongside continuous support and the
  initial5calls/2mice count. These thresholds describe support, not biological truth.
- Pairwise template cosine AND native-time execution-union Jaccard. Similar
  templates with different occurrence sets and overlapping executions with
  dissimilar templates remain distinguishable.
- Seed matching retains each family pair's cosine and execution overlap. Both
  phase-locked cosine and +/-1stage shifts are reported. Shifted matches keep
  full template norms, so a one-stage overlap cannot score as a complete match.
- Adjacent-K split/merge tables retain every parent/child edge, directional
  time-containment and template similarity. There is no forced one-to-one family
  correspondence across K or automatically accepted biological genealogy.

Three transparent shortlist views per vocabulary: time-coverage leader,
stable-family leader and compact K10coverage reference, deduplicated. Stable-family
selection uses the five strongest families' mean matched-template agreement across
other seeds; full distributions are also retained. These are verification
candidates, not a single weighted winner score or qualitative acceptance.

## Targeted order nulls

For each shortlisted model, freeze W and the original detection thresholds.
Infer H for real and null inputs with the same80updates and seed101. This avoids
comparing jointly fitted real H against differently optimised null H.
19paired surrogates/type are shared across shortlisted K/lambda models within
each vocabulary. These are descriptive diagnostics, not significance tests.

Frequency control permutes bout tokens within each valid block, preserving exact
bout counts. It can create adjacent repeats and is deliberately a weaker
first-order-breaking control; no second RLE is applied to alter the counts.
First-order control randomises Euler trails within each block, preserving exact
directed transition counts, endpoints and token frequencies. It consequently
preserves the no-self-transition constraint where the original RLE has it.
This procedure is not claimed to sample Euler trails uniformly; rigid/short
blocks may have little reorderable structure. Changed-token fractions are logged.

Native support/padding and duration slots stay fixed. Bout-count/coverage outcomes
are primary null diagnostics; null time coverage is conditional on those slots,
not a test preserving state-duration associations or real physical kinematics.
Failure to exceed a first-order null does not invalidate a coherent behavioural
family: it indicates its sequence statistics may be explained by transitions.

## Execution and products

Output root:
`/home/fsp585/Desktop/FiberphotometryCode/.worktrees/karen-seqnmf-delivery/outputs/karen_pooled_capacity_20261015`.

```bash
tail -f /home/fsp585/Desktop/FiberphotometryCode/.worktrees/karen-seqnmf-delivery/outputs/karen_pooled_capacity_20261015/run.log
```

Detached `nohup setsid timeout` job3734761,driver3734762,worker3734794.
RTX4090/float32/highest matmul precision,2numerical threads,8GiB RSSguard,
25%GPU allocation fraction,80GiB disk floor,4GiB output guard,3hour timeout.
Frozen `capacity_recipe.json` records input/base/source identities and scope;
results map every K/lambda/seed to its model/call SHA256 and directory.

Outputs:results.csv,capacity_metrics.csv,family_metrics.csv,redundancy_pairs.csv,
seed_family_matching.csv,split_merge_edges.csv.gz,null_shortlist.csv,
order_null_results.csv,order_null_preservation.csv,order_null_family_comparison.csv,
capacity_curves_*.png/pdf and READ_ME.md. Models/empirical tables remain ignored.
No new video generation or qualitative labels are implied by completion.

Checks before/alongside execution:12focused tests passed, including exact null
transition/frequency preservation, padding, physical union/containment, partial
shift penalties and historical boundary/update checks. FullCPU858passed/6existing
deprecation warnings. Lint3078reviewed/no new debt; maps2regenerated/146checked;
CLI plan/help read-only. No shared-library behaviour/debt inventory changed.

Next after completed diagnostics: inspect the capacity/coherence/coverage/stability
tradeoff, retain a compact and/or finer catalogue as justified, then prepare the
agreed p100/P80/random movies for targeted qualitative labelling. Do not infer
the largest K or lowest reconstruction error is automatically the best catalogue.

## Completed checkpoint

All60configurations and diagnostics finished. `final_audit.json` verifies model/
call hashes, parameters/shapes, finite arrays and native durations for862829
accepted calls/840factors. Supplementary grouped K18/lambda.3/seed43 null check
(selected after capacity curves, exploratory) adds a seventh candidate without
refitting. Seven models each have19surrogates/type and paired re-inferred realH.

At eachK, maximum physical time coverage across the six lambda/seed settings:

| Vocabulary | K6 | K10 | K14 | K18 | K22 |
| --- | --- | --- | --- | --- | --- |
| Original100 | 4.1% | 8.3% | 11.4% | 12.8% | 18.0% |
| Grouped69 | 13.8% | 17.3% | 23.2% | 31.2% | 35.0% |

These are candidate maxima, not method averages or proof thatK22is best. Seed
variation remains substantial and no clear plateau is established byK22. Accepted
matching errors are token-template measures, not independent physical coherence.
First-order null bout coverage is often close to real; frequency controls alone
cannot establish higher-order grammar. Descriptive null ranges do not justify
significance claims.

Text-first inspection uses the exact Karen raw100 and grouped69annotation CSVs
and actual accepted observed paths, with raw IDs/unknown states preserved.
`HUMAN_LABELLED_THEMES.md` and `human_labelled_families.csv` under the output
root support review before video work. Human descriptions are provisional,
including existing grooming subtype conventions, and label coherence is partly
built into the grouped input. All models use fixed three-bout templates; longer
variable-length routines have not been learned by this capacity sweep.
