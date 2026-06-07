import os
import cv2
import numpy as np

INPUT_DIR = r"C:\Users\chiti\Downloads\OcuScan project\data\STARE\train"
OUTPUT_DIR = r"C:\Users\chiti\Downloads\OcuScan project\results\stare_preprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".ppm")]

print(f"Found {len(files)} images. Starting preprocessing...")

for i, fname in enumerate(files):
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None:
        print(f"Skipping {fname} - could not read")
        continue

    # 1. Green channel extraction
    green = img[:, :, 1]

    # 2. Illumination correction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(green, background)

    # 3. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    # 4. Noise smoothing
    final = cv2.GaussianBlur(enhanced, (5, 5), 0)

    out_path = os.path.join(OUTPUT_DIR, fname.replace(".ppm", "_pre.png"))
    cv2.imwrite(out_path, final)
    print(f"[{i+1}/{len(files)}] Done: {fname}")

print("Preprocessing completed. Check results/stare_preprocessed")
