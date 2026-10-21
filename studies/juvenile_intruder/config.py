from studies.session_building.jonathan.config import JonathanConfig

# Range to keep (everything outside this is pre-excluded)
VALID_FRAME_RANGE = (50000, 250000)

# Custom config with relaxed regex for raw merged videos
# Filenames like: T1_4.8.e.e_Arena1and2_LeftTopCamVid_2026-01-20T15_07_45.avi
# We make the token optional/missing since raw videos don't have it.
JCFG = JonathanConfig(
    video_filename_regex = (
        r"^(?:.*_)?(?P<camset>(?P<arena>Arena[0-9and]+)_[^_]+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})"
        r"(?:_(?P<token>.+?))?\.(?P<ext>avi|mp4)$"
    )
)

# Monkey-patch token_to_chamber to handle Arena-specific files with no token
_orig_t2c = JCFG.token_to_chamber
def patched_t2c(camset: str, token: str) -> str:
    # camset now captures "ArenaX_Name", so strip Arena prefix to check original logic
    stripped_camset = camset
    if camset and "_" in camset and camset.startswith("Arena"):
        parts = camset.split("_", 1)
        if len(parts) > 1:
            stripped_camset = parts[1]

    # Try original Logic
    res = _orig_t2c(stripped_camset, token)

    # Debug logging
    # print(f"[DEBUG-PATCH] camset='{camset}', token='{token}', stripped='{stripped_camset}', orig_res='{res}'")

    # Original returns token (empty) if no match, so check if it's a valid chamber name if possible
    # But usually it returns the chamber name. If it fails it returns 'token' which might be None.
    if res and res != token:
        # print(f"  -> Returning orig_res: {res}")
        return res
    if res and JCFG.is_canon_chamber(res):
        # print(f"  -> Returning canon_res: {res}")
        return res

    # Fallback: Inference from Arena prefix in camset
    c_lower = (camset or "").lower()
    if "arena1_" in c_lower:
        # print("  -> Inferred top_left")
        return "top_left"
    if "arena2_" in c_lower: return "bottom_left"
    if "arena3_" in c_lower: return "bottom_right" # Assumed order
    if "arena4_" in c_lower: return "top_right"    # Assumed order

    # print(f"  -> Returning fallback res: {res}")
    return res

JCFG.token_to_chamber = patched_t2c
