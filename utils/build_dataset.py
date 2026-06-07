import pandas as pd
import os

PROJECT_PATH = r"C:\Users\chiti\OneDrive\Documents\OcuScan project"

disc_file = os.path.join(PROJECT_PATH,"results","disc_features.csv")
vessel_file = os.path.join(PROJECT_PATH,"results","vessel_features.csv")
output_file = os.path.join(PROJECT_PATH,"results","retina_dataset.csv")

disc_df = pd.read_csv(disc_file)
vessel_df = pd.read_csv(vessel_file)

# remove extensions
disc_df["id"] = disc_df["id"].str.replace(".png","",regex=False)
disc_df["id"] = disc_df["id"].str.replace(".tif","",regex=False)

vessel_df["image"] = vessel_df["image"].str.replace(".png","",regex=False)
vessel_df["image"] = vessel_df["image"].str.replace(".tif","",regex=False)

vessel_df = vessel_df.rename(columns={"image":"id"})

dataset = pd.merge(disc_df, vessel_df, on="id", how="inner")

dataset.to_csv(output_file,index=False)

print("Dataset created")
print("Total rows:", len(dataset))
print("Saved to:",output_file)
