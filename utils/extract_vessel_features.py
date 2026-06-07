import cv2
import numpy as np
import pandas as pd
import os

PROJECT_PATH = r"C:\Users\chiti\OneDrive\Documents\OcuScan project"

INPUT_FOLDER = os.path.join(PROJECT_PATH,"results","vessel_maps")
OUTPUT_FILE = os.path.join(PROJECT_PATH,"results","vessel_features.csv")


def skeletonize(img):
    img = img.copy()
    skel = np.zeros(img.shape,np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    while True:
        eroded = cv2.erode(img,kernel)
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        if cv2.countNonZero(img)==0:
            break

    return skel


results = []

for file in os.listdir(INPUT_FOLDER):

    if not file.endswith((".png",".jpg",".tif",".jpeg",".bmp")):
        continue

    path = os.path.join(INPUT_FOLDER,file)

    img = cv2.imread(path,0)

    if img is None:
        continue

    total_pixels = img.shape[0] * img.shape[1]

    vessel_pixels = np.sum(img > 0)

    vessel_density = vessel_pixels / total_pixels

    # Skeleton for vessel length
    skel = skeletonize(img)

    vessel_length = cv2.countNonZero(skel)

    results.append({
        "image": file,
        "vessel_pixels": vessel_pixels,
        "vessel_density": vessel_density,
        "vessel_length": vessel_length
    })


df = pd.DataFrame(results)

df.to_csv(OUTPUT_FILE,index=False)

print("Vessel features extracted")
print("Saved to:",OUTPUT_FILE)
