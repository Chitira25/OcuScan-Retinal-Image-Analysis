import os
import cv2
import numpy as np

RAW_DIR    = r"C:\Users\chiti\Downloads\OcuScan project\data\STARE\train"
VESSEL_DIR = r"C:\Users\chiti\Downloads\OcuScan project\results\stare_vessels"
OUTPUT_DIR = r"C:\Users\chiti\Downloads\OcuScan project\results\stare_optic_disc"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".ppm")]
print(f"Found {len(files)} STARE images.")

for i, fname in enumerate(files):
    img_color = cv2.imread(os.path.join(RAW_DIR, fname))
    if img_color is None:
        print(f"  Skipping {fname}")
        continue

    green = img_color[:, :, 1]
    green = cv2.normalize(green, None, 0, 255, cv2.NORM_MINMAX)

    vessel_path = os.path.join(VESSEL_DIR, fname.replace(".ppm", "_vessel.png"))
    vessel = cv2.imread(vessel_path, 0)
    if vessel is None:
        print(f"  Missing vessel map for {fname}, skipping")
        continue

    density = cv2.GaussianBlur(vessel, (61, 61), 0)
    bright  = cv2.GaussianBlur(green,  (61, 61), 0)

    score = cv2.normalize(
        bright.astype(np.float32) * density.astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, thresh = cv2.threshold(score, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  No disc found for {fname}")
        continue

    best = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(green)
    cv2.drawContours(mask, [best], -1, 255, -1)

    out_path = os.path.join(OUTPUT_DIR, fname.replace(".ppm", "_disc.png"))
    cv2.imwrite(out_path, mask)
    print(f"  [{i+1}/{len(files)}] Done: {fname}")

print("Optic disc detection completed. Check results/stare_optic_disc")
