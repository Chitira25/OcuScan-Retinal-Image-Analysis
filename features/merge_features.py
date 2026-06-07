import pandas as pd

vessel_csv = r"results\vessel_features.csv"
disc_csv = r"results\disc_features.csv"

vdf = pd.read_csv(vessel_csv)
ddf = pd.read_csv(disc_csv)

# normalize image IDs
vdf["id"] = vdf["image"].str.replace("_vessel.png", "", regex=False)
ddf["id"] = ddf["image"].str.replace("_disc.png", "", regex=False)

merged = pd.merge(vdf, ddf, on="id", how="inner")

merged = merged.drop(columns=["image_x", "image_y"])

out_path = r"results\merged_features.csv"
merged.to_csv(out_path, index=False)

print("Merged features saved to:", out_path)
print(merged.head())
