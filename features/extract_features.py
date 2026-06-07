import os
import cv2
import numpy as np
import csv
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

VESSEL_DIR = r"C:\Users\chiti\OneDrive\Desktop\Documents\OcuScan project\results\vessels"
OUT_CSV = r"C:\Users\chiti\OneDrive\Desktop\Documents\OcuScan project\results\vessel_features.csv"

rows = []
rows.append(["image", "vessel_density", "vessel_length_px", "mean_tortuosity_proxy"])

files = [f for f in os.listdir(VESSEL_DIR) if f.endswith("_vessel.png")]

for fname in files:
    mask = cv2.imread(os.path.join(VESSEL_DIR, fname), cv2.IMREAD_GRAYSCALE)
    _, bin_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # Vessel density
    density = bin_mask.sum() / bin_mask.size

    # Skeletonize to get length
    skeleton = skeletonize(bin_mask.astype(bool))
    vessel_length = skeleton.sum()

    # Tortuosity proxy: number of connected components / length (rough indicator)
    labeled = label(skeleton)
    components = len(regionprops(labeled))
    tortuosity_proxy = components / (vessel_length + 1e-6)

    rows.append([fname, density, vessel_length, tortuosity_proxy])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Feature extraction done. Saved:", OUT_CSV)
