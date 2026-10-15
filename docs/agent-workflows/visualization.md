# KPMS media with explicit frame mappings

Prefer results/coordinate artifacts over loading a checkpoint when rendering does not need latent reconstruction. The canonical entry points are `fpcode.cli.render`, `fpcode.workflows.rendering`, `fpcode.viz.grid_movies` and `fpcode.viz.trajectories`.

The rendering CLI accepts results HDF5 and a YAML manifest. Paths inside the manifest are relative to the manifest directory. Supply real FPS and mapping policy explicitly; the code has historical defaults that are not proof of the source clock.

```yaml
fps: 40                 # Example only: choose the actual label timeline rate.
sampling_mode: random   # Or density; it has additional population requirements.
coordinates:
  recording_1: coordinates.npy
video_paths:
  recording_1: recording.avi
video_frame_indexes:
  recording_1: native_video_frames.npy
render_options:
  pre: 1.0
  post: 2.0
  min_duration: 3
  overlay_keypoints: false
```

Results must contain each recording's integer syllable sequence. Coordinates have `(label_frames, keypoints, 2 or 3)` shape. Grid movies also need matched centroid `(label_frames, 2)` and heading `(label_frames,)` arrays. Every selected recording needs the required mappings. Validate the intended recording set; silently intersecting keys can exclude scientific inputs.

VFI values in this workflow name **native source-video frames**, one per label/coordinate row. They are not automatically model-row indexes. Equal FPS does not prove that the clocks share an origin. Read acquisition/model metadata and verify offset/rate mapping against known frames. Do not multiply VFI by two, repeat labels or infer clock units from maximum index/recording length without that evidence. Negative/missing/duplicate mappings need an explicit supported policy; the current render loader requires increasing nonnegative indexes.

```bash
python -B -m fpcode.cli.doctor --json --require render \
  --input /path/to/results.h5 --input /path/to/render.yaml
python -B -m fpcode.cli.render --results /path/to/results.h5 \
  --manifest /path/to/render.yaml --type trajectories --syllables 0 \
  --output-dir /path/to/preview --dry-run --no-notify
```

Use `--type grid_movies` for movies and a requested `--grid-size`; dry-run validates eligible bout/population requirements. Dry-run reads the named results and coordinate arrays but does not encode media or fit a model. Bound the preview's inputs and selected syllables explicitly; the CLI has no generic duration/sample-limit flag.

Density selection differs scientifically from random selection and needs an adequate reference population. Do not substitute a mode when requested inputs are insufficient; diagnose the unmet requirement. `render_options.overlay_keypoints` selects overlays. Check a requested preview for frame alignment, labels, artifact count/nonzero size and capture/encoding validity before a large render.

If checkpoint reconstruction is actually necessary, choose the project's trusted backend environment and device before JAX initialization, preserve the model-owned configuration, and pass reconstructed arrays into the same canonical renderer. Session pickles require intentional reconstruction through compatible namespaces; never unpickle them during metadata-only inventory. Backend-version-specific adapters should remain scoped/restorable rather than global notebook patches.

Record input/clock policy, selection mode, output location and limitations. Use objective state descriptions. See [validation selection](validation.md) and [current task state](../tasks/CURRENT.md).
