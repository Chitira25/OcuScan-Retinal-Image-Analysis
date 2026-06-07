import os
import cv2
import numpy as np

INPUT_DIR  = r"C:\Users\chiti\Downloads\OcuScan project\results\stare_preprocessed"
OUTPUT_DIR = r"C:\Users\chiti\Downloads\OcuScan project\results\stare_vessels"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_pre.png")]
print(f"Found {len(files)} preprocessed images.")

for i, fname in enumerate(files):
    img = cv2.imread(os.path.join(INPUT_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Skipping {fname}")
        continue

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    thresh = cv2.adaptiveThreshold(
        tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, -2
    )

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small)

    out_path = os.path.join(OUTPUT_DIR, fname.replace("_pre.png", "_vessel.png"))
    cv2.imwrite(out_path, cleaned)
    print(f"  [{i+1}/{len(files)}] Done: {fname}")

print("Vessel segmentation completed. Check results/stare_vessels")
