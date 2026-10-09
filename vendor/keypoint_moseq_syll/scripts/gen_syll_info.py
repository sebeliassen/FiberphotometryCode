import os

import pandas as pd

model_dir = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/flipped_double_project/double_flip_activity_k100k"
out_csv = os.path.join(model_dir, "syll_info.csv")

data = []
for i in range(100):
    movie_path = os.path.join(model_dir, f"grid_movies/syllable{i}.mp4")
    data.append({
        "syllable": i,
        "label": "",
        "short_description": "",
        "movie_path": movie_path if os.path.exists(movie_path) else ""
    })

df = pd.DataFrame(data)

overrides = {
    13: {"label": "right turn", "short_description": "partial/45 deg"},
    18: {"label": "left turn", "short_description": ""},
    20: {"label": "look right", "short_description": ""},
    28: {"label": "stand down / look left", "short_description": ""},
    36: {"label": "look left", "short_description": ""},
    40: {"label": "", "short_description": ""},
    41: {"label": "groom right", "short_description": ""},
    42: {"label": "groom left", "short_description": ""}
}

for syll, info in overrides.items():
    idx = df[df["syllable"] == syll].index
    if not idx.empty:
        df.loc[idx, "label"] = info["label"]
        df.loc[idx, "short_description"] = info["short_description"]

df.to_csv(out_csv, index=False)
print(f"Generated {out_csv}")
