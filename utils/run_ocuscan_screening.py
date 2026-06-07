import cv2
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tkinter import Tk, filedialog

# ----------------------------------
# Train model using existing dataset
# ----------------------------------

data = pd.read_csv("results/retina_dataset.csv")

X = data.drop(["id","label"], axis=1)
y = data["label"]

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X,y)

print("Model trained successfully")


# ----------------------------------
# Select retinal image
# ----------------------------------

Tk().withdraw()

file_path = filedialog.askopenfilename(
    title="Select Retinal Image",
    filetypes=[("Image Files","*.tif *.png *.jpg")]
)

print("Selected image:", file_path)

img = cv2.imread(file_path)


# ----------------------------------
# Vessel Segmentation (clean method)
# ----------------------------------

green = img[:,:,1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(green)

blur = cv2.GaussianBlur(enhanced,(5,5),0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)

vessel = cv2.adaptiveThreshold(
    tophat,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    15,
    -2
)

kernel_small = np.ones((3,3),np.uint8)
vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, kernel_small)


# Vessel Thickness Map
dist_transform = cv2.distanceTransform(vessel, cv2.DIST_L2, 5)

thickness_norm = cv2.normalize(
    dist_transform,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

thickness_color = cv2.applyColorMap(
    thickness_norm,
    cv2.COLORMAP_JET
)
# ----------------------------------
# Vessel Features
# ----------------------------------

vessel_pixels = np.sum(vessel == 255)
total_pixels = vessel.size

vessel_density = vessel_pixels / total_pixels
vessel_length = vessel_pixels


# ----------------------------------
# Optic Disc Estimation
# ----------------------------------

disc_radius = 60
disc_center_x = img.shape[1]//2
disc_center_y = img.shape[0]//2
disc_area = np.pi * disc_radius**2

# Optic Disc Heatmap
heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

cv2.circle(
    heatmap,
    (disc_center_x, disc_center_y),
    disc_radius,
    255,
    -1
)

heatmap = cv2.GaussianBlur(heatmap,(101,101),0)

heatmap_color = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_HOT
)

heatmap_overlay = cv2.addWeighted(
    img,
    0.7,
    heatmap_color,
    0.3,
    0
)


# ----------------------------------
# Feature Vector
# ----------------------------------

features = np.array([[
    disc_center_x,
    disc_center_y,
    disc_radius,
    disc_area,
    vessel_pixels,
    vessel_density,
    vessel_length
]])

prediction = model.predict(features)[0]
confidence = model.predict_proba(features)[0].max()


# ----------------------------------
# Create Vessel Overlay
# ----------------------------------

overlay = img.copy()

overlay[vessel == 255] = [0,255,0]   # green vessels


# optic disc circle
cv2.circle(
    overlay,
    (disc_center_x, disc_center_y),
    disc_radius,
    (255,0,0),
    2
)


# abnormality highlight
if prediction == 1:
    cv2.rectangle(
        overlay,
        (10,10),
        (overlay.shape[1]-10, overlay.shape[0]-10),
        (0,0,255),
        4
    )

# Abnormality Zone Detection
if prediction == 1:

    abnormal_mask = cv2.GaussianBlur(vessel,(31,31),0)

    _, abnormal_mask = cv2.threshold(
        abnormal_mask,
        50,
        255,
        cv2.THRESH_BINARY
    )

    abnormal_mask = cv2.applyColorMap(
        abnormal_mask,
        cv2.COLORMAP_AUTUMN
    )

    overlay = cv2.addWeighted(
        overlay,
        0.8,
        abnormal_mask,
        0.4,
        0
    )


# ----------------------------------
# Prepare Images
# ----------------------------------

img_display = cv2.resize(overlay, (520,520))

vessel_display = cv2.resize(vessel, (520,520))
vessel_display = cv2.cvtColor(vessel_display, cv2.COLOR_GRAY2BGR)


# ----------------------------------
# Create Professional Dashboard
# ----------------------------------

dashboard = np.zeros((720,1200,3), dtype=np.uint8)
dashboard[:] = (25,25,25)   # dark theme background


# HEADER BAR
cv2.rectangle(dashboard,(0,0),(1200,60),(40,40,40),-1)

cv2.putText(
    dashboard,
    "OCUSCAN RETINAL ANALYSIS SYSTEM",
    (350,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0,255,200),
    2
)


# IMAGE PANELS
cv2.rectangle(dashboard,(40,100),(560,620),(60,60,60),2)
cv2.rectangle(dashboard,(640,100),(1160,620),(60,60,60),2)

dashboard[100:620,40:560] = img_display
dashboard[100:620,640:1160] = vessel_display


# PANEL TITLES
cv2.putText(
    dashboard,
    "Retinal Image + Vessel Overlay",
    (150,90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,255,255),
    2
)

cv2.putText(
    dashboard,
    "Extracted Vessel Network",
    (780,90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,255,255),
    2
)


# ANALYSIS PANEL
cv2.rectangle(dashboard,(40,640),(1160,700),(40,40,40),-1)


result_text = "NORMAL" if prediction == 0 else "ABNORMAL"
color = (0,255,0) if prediction == 0 else (0,0,255)

cv2.putText(
    dashboard,
    f"Diagnosis: {result_text}",
    (60,680),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    color,
    2
)

cv2.putText(
    dashboard,
    f"Vessel Density: {vessel_density:.4f}",
    (350,680),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255,255,255),
    2
)

cv2.putText(
    dashboard,
    f"Vessel Pixels: {vessel_pixels}",
    (650,680),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255,255,255),
    2
)

cv2.putText(
    dashboard,
    f"Model Confidence: {confidence:.2f}",
    (900,680),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255,255,255),
    2
)


# ----------------------------------
# Display Dashboard
# ----------------------------------

cv2.imshow("OcuScan Dashboard", dashboard)

cv2.waitKey(0)
cv2.destroyAllWindows()
