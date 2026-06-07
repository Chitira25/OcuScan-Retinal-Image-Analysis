import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

PROJECT = r"C:\Users\chiti\Downloads\OcuScan project"

INPUT_DIR       = os.path.join(PROJECT, "results", "stare_preprocessed")
VESSEL_DIR      = os.path.join(PROJECT, "results", "stare_vessels")
DISC_DIR        = os.path.join(PROJECT, "results", "stare_optic_disc")
DRIVE_CSV       = os.path.join(PROJECT, "results", "retina_dataset.csv")
STARE_CSV       = os.path.join(PROJECT, "results", "stare_dataset.csv")
COMBINED_CSV    = os.path.join(PROJECT, "results", "combined_dataset.csv")
MODEL_OUT       = os.path.join(PROJECT, "results", "model_combined.pkl")

os.makedirs(VESSEL_DIR, exist_ok=True)
os.makedirs(DISC_DIR,   exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — VESSEL SEGMENTATION
# ─────────────────────────────────────────────
print("\n=== STEP 1: Vessel Segmentation ===")

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith("_pre.png")]
print(f"Found {len(files)} preprocessed STARE images")

for i, fname in enumerate(files):
    img = cv2.imread(os.path.join(INPUT_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Skipping {fname}")
        continue

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_map = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    out_path = os.path.join(VESSEL_DIR, fname.replace("_pre.png", "_vessel.png"))
    cv2.imwrite(out_path, vessel_map)
    print(f"  [{i+1}/{len(files)}] Vessel done: {fname}")

print("Vessel segmentation complete.")

# ─────────────────────────────────────────────
# STEP 2 — OPTIC DISC DETECTION
# ─────────────────────────────────────────────
print("\n=== STEP 2: Optic Disc Detection ===")

for i, fname in enumerate(files):
    img = cv2.imread(os.path.join(INPUT_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    disc_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(disc_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(disc_img, (x, y), 2, (0, 0, 255), 3)

    out_path = os.path.join(DISC_DIR, fname.replace("_pre.png", "_disc.png"))
    cv2.imwrite(out_path, disc_img)
    print(f"  [{i+1}/{len(files)}] Disc done: {fname}")

print("Optic disc detection complete.")

# ─────────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION
# ─────────────────────────────────────────────
print("\n=== STEP 3: Feature Extraction ===")

rows = []

for fname in files:
    image_id = fname.replace("_pre.png", "")

    # Vessel features
    vessel_path = os.path.join(VESSEL_DIR, fname.replace("_pre.png", "_vessel.png"))
    vessel_map = cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE)

    if vessel_map is None:
        vessel_pixels  = 0
        vessel_density = 0.0
        vessel_length  = 0
    else:
        total_pixels   = vessel_map.shape[0] * vessel_map.shape[1]
        vessel_pixels  = int(np.sum(vessel_map > 0))
        vessel_density = round(vessel_pixels / total_pixels, 6)
        vessel_length  = vessel_pixels  # approximation

    # Disc features from preprocessed image
    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    disc_center_x = -1
    disc_center_y = -1
    disc_radius   = 0
    disc_area     = 0

    if img is not None:
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=50, param2=30,
            minRadius=20, maxRadius=100
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            disc_center_x = int(x)
            disc_center_y = int(y)
            disc_radius   = int(r)
            disc_area     = int(np.pi * r * r)

    rows.append({
        "id":             image_id,
        "disc_center_x":  disc_center_x,
        "disc_center_y":  disc_center_y,
        "disc_radius_px": disc_radius,
        "disc_area_px":   disc_area,
        "vessel_pixels":  vessel_pixels,
        "vessel_density": vessel_density,
        "vessel_length":  vessel_length,
        "label":          0
    })
    print(f"  Features extracted: {image_id}")

stare_df = pd.DataFrame(rows)
stare_df.to_csv(STARE_CSV, index=False)
print(f"STARE features saved to {STARE_CSV}")
print(f"Total STARE images processed: {len(stare_df)}")

# ─────────────────────────────────────────────
# STEP 4 — MERGE WITH DRIVE DATASET
# ─────────────────────────────────────────────
print("\n=== STEP 4: Merging DRIVE + STARE ===")

drive_df    = pd.read_csv(DRIVE_CSV)
combined_df = pd.concat([drive_df, stare_df], ignore_index=True)
combined_df.to_csv(COMBINED_CSV, index=False)
print(f"DRIVE rows:    {len(drive_df)}")
print(f"STARE rows:    {len(stare_df)}")
print(f"Combined rows: {len(combined_df)}")
print(f"Saved to {COMBINED_CSV}")

# ─────────────────────────────────────────────
# STEP 5 — RETRAIN MODEL
# ─────────────────────────────────────────────
print("\n=== STEP 5: Retraining Classifier ===")

FEATURES = ["disc_center_x", "disc_center_y", "disc_radius_px",
            "disc_area_px", "vessel_pixels", "vessel_density", "vessel_length"]

df = combined_df.dropna(subset=FEATURES + ["label"])
X  = df[FEATURES]
y  = df["label"]

if y.nunique() < 2:
    print("WARNING: Only one class in dataset — adding dummy row for training.")
    dummy = pd.DataFrame([{f: 0 for f in FEATURES}])
    dummy["label"] = 1
    X = pd.concat([X, dummy[FEATURES]], ignore_index=True)
    y = pd.concat([y, dummy["label"]], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to {MODEL_OUT}")
print("\n=== ALL STEPS COMPLETE ===")
print("Files created:")
print(f"  {STARE_CSV}")
print(f"  {COMBINED_CSV}")
print(f"  {MODEL_OUT}")
