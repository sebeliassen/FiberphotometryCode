# Configuration ownership

| Location | Consumer and purpose |
| --- | --- |
| `training/*.yaml` | Declarative runner profiles and the installed training CLI |
| `training/*.yml` | Retained legacy queue manifests, with original field names and values |
| `fusion/*.yaml` | Fusion workflows; explicit inputs or the named legacy compatibility profile |
| `photometry/*.yaml` | Explicit time/alignment policy examples for the Python photometry workflows |
| `backup.yaml` | Snapshot planning, exclusions, verification and restoration |

The two historical flat files `cohort_static_21kp.yml` and
`tandem_cohort_50ar_200slds.yml` are compatibility symlinks to their unchanged
contents in `training/`. Historical callers and reports can keep their paths.
Legacy queue manifests are not interchangeable with the installed CLI schema;
for example the static queue's `save_every_n` spelling remains historical.

The installed training CLI consumes already-preprocessed data: `data_path`
points to `preprocessed_data.pkl`. It does not prepare tracking data merely from
`data_dir`. Validate a request with `fpcode-train --config <path> --dry-run` before
an explicit training call. Effective run settings are recorded with its outputs.

`photometry/event_windows.yaml` is a small policy example, not a cohort or a
complete executable pipeline. A Python workflow caller loads it, builds signal
and event contracts with the declared unit/clock, and passes `offsets` plus
`alignment` to `fpcode.workflows.photometry.event_photometry`. In this example
linear interpolation is limited to gaps of 0.10 seconds and never extrapolates
outside the source recording. Real studies choose these settings explicitly.

Model-owned `config.yml`, PCA files and scientific artifacts retain their
existing locations. New examples here do not change historical study policies.
