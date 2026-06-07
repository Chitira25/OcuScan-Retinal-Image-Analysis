import cv2
import numpy as np
import pandas as pd
import os

PROJECT_PATH = r"C:\Users\chiti\OneDrive\Documents\OcuScan project"

INPUT_FOLDER = os.path.join(PROJECT_PATH,"preprocessing","output","clahe")
OUTPUT_OVERLAY = os.path.join(PROJECT_PATH,"results","disc_overlays")
OUTPUT_CSV = os.path.join(PROJECT_PATH,"results","disc_features.csv")

os.makedirs(OUTPUT_OVERLAY, exist_ok=True)

results = []

print("Reading images from:", INPUT_FOLDER)

for file in os.listdir(INPUT_FOLDER):

    if not file.lower().endswith((".png",".jpg",".jpeg",".tif",".bmp")):
        continue

    path = os.path.join(INPUT_FOLDER,file)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img,(565,584))

    green = img[:,:,1]

    # smooth image
    blur = cv2.GaussianBlur(green,(11,11),0)

    # threshold bright regions
    _,th = cv2.threshold(blur,220,255,cv2.THRESH_BINARY)

    # morphological cleanup
    kernel = np.ones((25,25),np.uint8)
    th = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel)

    contours,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    best_area = 0

    for c in contours:

        area = cv2.contourArea(c)

        if area < 2000:
            continue

        (x,y),r = cv2.minEnclosingCircle(c)

        if r < 40 or r > 120:
            continue

        if area > best_area:
            best_area = area
            best_circle = (int(x),int(y),int(r))

    # ---- HANDLE FAILURE CASE ----
    if best_circle is None:
        x = -1
        y = -1
        r = -1
    else:
        x,y,r = best_circle

    overlay = img.copy()

    if r > 0:
        cv2.circle(overlay,(x,y),r,(0,255,0),3)
        cv2.circle(overlay,(x,y),3,(0,0,255),-1)

    cv2.imwrite(os.path.join(OUTPUT_OVERLAY,file),overlay)

    results.append({
        "id": file,
        "disc_center_x": x,
        "disc_center_y": y,
        "disc_radius_px": r,
        "disc_area_px": np.pi*r*r if r > 0 else -1
    })

    print("Processed:",file)

df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV,index=False)

print("\nOptic disc detection complete")
print("Saved to:",OUTPUT_CSV)
print("Total images processed:",len(df))
