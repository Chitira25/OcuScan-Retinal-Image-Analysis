import pandas as pd
import os

PROJECT_PATH = r"C:\Users\chiti\OneDrive\Documents\OcuScan project"

disc = pd.read_csv(os.path.join(PROJECT_PATH,"results","disc_features.csv"))
vessel = pd.read_csv(os.path.join(PROJECT_PATH,"results","vessel_features.csv"))

print("DISC IDs:")
print(disc["id"].tolist())

print("\nVESSEL IDs:")
print(vessel["image"].tolist())
