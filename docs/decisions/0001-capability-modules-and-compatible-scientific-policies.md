# Capability modules and compatible scientific policies

Date: 2026-10-09. Status: implemented foundation; consumer migration continues.

The shared library is organized by capabilities: contracts, adapters, alignment,
tracking, photometry, behavior, KPMS, statistics, visualization and workflows.
Studies retain cohort definitions, exclusions, hypotheses and scientific choices.
CLI modules load configuration, call workflows and record effective settings.

Recursive modularity means decomposing a capability when it contains distinct
responsibilities. It does not require a CLI, README and YAML file in every leaf
directory. Algorithms and contracts should remain independently usable.

Historical behavior is a compatibility policy, not an implicit default for new
workflows. Row-position interpolation, endpoint filling, sparse/dense syllable
timelines, confidence/presence rules and smoothing variants remain distinguishable.
New strict APIs require explicit units, clocks, tolerances and gap limits.

Compatibility aliases preserve historical imports and object identities. Their
removal requires a consumer inventory and representative historical pickle checks.
The default functional suite uses small deterministic inputs. Empirical and GPU
checks are separate gates; passing functional tests does not certify scientific
reproduction or accelerated sampler equivalence.
