# Ground Truth VFI Architecture & Context

**Goal**: Maximize yield for future development by documenting the "Ground Truth VFI" paradigm and the verification results.

## 1. Core Paradigm: "Ground Truth VFI"
*   **Definition**: The Video Frame Index (VFI) (`video_frame_indexes.npz`) **ALWAYS** refers to the absolute frame numbers (0, 1, 2...) of the original, high-frequency (80Hz) video.
*   **Invariant**: Even if the model is trained on downsampled data (e.g., 40Hz), the stored VFI maps those downsampled steps back to their original absolute positions (e.g., `0, 2, 4...`).
*   **No "Rate Detection"**: Downstream analysis scripts must **NEVER** attempt to guess a downsampling rate to multiply indices. They should trust the VFI as absolute truth.

## 2. Legacy Data Handling
*   **The Artifact**: Old `.pkl` files and `.npz` files produced before Jan 2026 often contain "Relative/Downsampled" indices (0, 1, 2... for a 40Hz stream).
*   **The Solution**: `syllable_generation/analysis/utils/legacy_support.py`.
    *   **Heuristic**: If `VFI_Max < (Session_Length / 1.5)`, it is flagged as Legacy.
    *   **Migration**: It is scaled on-the-fly (e.g., x2.0) to convert it to Ground Truth.
*   **Rule**: All legacy logic is isolated in `legacy_support.py`. Do not re-introduce "rate args" into main functions.

## 3. Analysis Pipeline Flow
The key script `attach_syllables_kpms.py` follows this strict flow to guarantee alignment:

1.  **Load Data**: Sessions (Photometry) + KPMS Results + VFI.
2.  **Migrate VFI**: `legacy_support.migrate_legacy_vfi()` ensures VFI is absolute Int64.
3.  **Align**: `sampling.build_aligned_dfs()`:
    *   Maps Results to VFI.
    *   **Densification**: Uses `ffill` (forward fill) to bridge gaps between VFI indices (e.g., filling index 1 between 0 and 2).
    *   **Full Coverage**: Forces the DataFrame index to be `0` to `Photometry_Length`.
    *   **Result**: `all_data_df` matches `photometry` 1:1.

## 4. Key Files & Responsibilities
| File | Responsibility |
| :--- | :--- |
| `syllabizers/.../utils/trimming.py` | **Source of Truth**. `filter_with_masks` calculates absolute indices during generation. |
| `analysis/utils/legacy_support.py` | **Back-Compat**. Detects and fixes old VFI files on read. |
| `analysis/utils/sampling.py` | **Alignment**. `build_aligned_dfs` handles densification and dataframe construction. |
| `analysis/attach_syllables_kpms.py` | **Orchestrator**. Wiring it all together. |

## 5. Verification Status
*   **Equivalence**: Verified on real Amph data (`verify_legacy_migration.py`) that the new pipeline produces **0 differences** in syllable assignments compared to the old legacy pipeline for overlapping frames.
*   **Coverage**: Verified that the new pipeline recovers frames at the start/end of sessions that were previously clipped, ensuring no data loss.

## 6. Critical Reminders for Future Agents
1.  **Do not add `downsample_rate`** arguments to analysis functions. If alignment looks wrong, check `legacy_support` logic, don't monkey-patch scaling.
2.  **Trust `all_data_df` length**. It is now guaranteed to match the photometry signal length.
3.  **Nullable Integers**: Use `Int64` (capital I) for syllable columns to gracefully handle the `NaN`s that naturally occur during aligning/padding.
