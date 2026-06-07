import os
import cv2
import numpy as np
import csv

DISC_DIR = r"C:\Users\chiti\OneDrive\Desktop\Documents\OcuScan project\results\optic_disc"
OUT_CSV = r"C:\Users\chiti\OneDrive\Desktop\Documents\OcuScan project\results\disc_features.csv"

rows = [["image", "disc_area_px", "disc_radius_px"]]

files = [f for f in os.listdir(DISC_DIR) if f.endswith("_disc.png")]

for fname in files:
    mask = cv2.imread(os.path.join(DISC_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    area = np.sum(bin_mask > 0)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
    else:
        radius = 0

    rows.append([fname, area, radius])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Disc features saved:", OUT_CSV)
