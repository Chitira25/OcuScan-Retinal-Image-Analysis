import cv2
import numpy as np
import os

# ===== PROJECT PATH =====
PROJECT_PATH = r"C:\Users\chiti\OneDrive\Documents\OcuScan project"

INPUT_FOLDER = os.path.join(PROJECT_PATH, "preprocessing", "output", "clahe")
OUTPUT_FOLDER = os.path.join(PROJECT_PATH, "results", "vessel_maps")

# Create output folder if missing
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check input folder
if not os.path.exists(INPUT_FOLDER):
    print("ERROR: CLAHE folder not found")
    print("Expected here:", INPUT_FOLDER)
    exit()

print("Reading images from:", INPUT_FOLDER)

for file in os.listdir(INPUT_FOLDER):

    if not file.lower().endswith((".png",".jpg",".jpeg",".tif",".bmp")):
        continue

    img_path = os.path.join(INPUT_FOLDER, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (565,584))

    # Use green channel (best for vessels)
    green = img[:,:,1]

    # Blur
    blur = cv2.GaussianBlur(green,(5,5),0)

    # Morphological TopHat to enhance vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)

    # Threshold
    _, vessels = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel_small = np.ones((3,3),np.uint8)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel_small)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, kernel_small)

    out_path = os.path.join(OUTPUT_FOLDER, file)

    cv2.imwrite(out_path, vessels)

    print("Processed:", file)

print("\nVessel segmentation finished")
print("Results saved in:", OUTPUT_FOLDER)
