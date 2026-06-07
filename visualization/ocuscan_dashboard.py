import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ---------------------------
# Load trained model
# ---------------------------

PROJECT    = r"C:\Users\chiti\Downloads\OcuScan project"
MODEL_PATH = os.path.join(PROJECT, "results", "model_combined.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model_loaded = True
except:
    model_loaded = False

# ---------------------------
# Globals
# ---------------------------

img_path    = None
img_display = None

# ---------------------------
# Window
# ---------------------------

root = tk.Tk()
root.title("OcuScan Clinical Dashboard")
root.geometry("1250x720")
root.configure(bg="#0b1a2a")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

# ---------------------------
# Title
# ---------------------------

tk.Label(root,
         text="OcuScan Clinical Retinal Analysis",
         font=("Segoe UI", 22, "bold"),
         fg="#00e5ff",
         bg="#0b1a2a").grid(row=0, column=0, columnspan=2, pady=(18, 2))

model_status = "✅  Model Loaded  |  Accuracy: 92.31%  |  DRIVE + STARE  |  260 samples" \
               if model_loaded else "❌  Model not found — using threshold fallback"

tk.Label(root,
         text=model_status,
         font=("Segoe UI", 9),
         fg="#22c55e" if model_loaded else "#ef4444",
         bg="#0b1a2a").grid(row=1, column=0, columnspan=2, pady=(0, 10))

# ---------------------------
# IMAGE PANEL
# ---------------------------

image_frame = tk.Frame(root, bg="#10273f", width=600, height=520)
image_frame.grid(row=2, column=0, padx=30, pady=10)
image_frame.grid_propagate(False)

tk.Label(image_frame,
         text="Retinal Image",
         font=("Segoe UI", 14, "bold"),
         bg="#10273f",
         fg="white").pack(pady=10)

image_label = tk.Label(image_frame, bg="black", fg="gray")
image_label.pack(fill="both", expand=True, pady=10)

# Placeholder with eye icon
placeholder = Image.new("RGB", (580, 480), color="#0a1520")
pixels = placeholder.load()
center_x, center_y = 290, 240

for x in range(580):
    for y in range(480):
        dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
        if 95 <= dist <= 100 or 50 <= dist <= 55:
            pixels[x, y] = (0, 229, 255)
        elif dist < 50:
            pixels[x, y] = (10, 21, 32)

placeholder_tk = ImageTk.PhotoImage(placeholder)
image_label.config(image=placeholder_tk)
image_label.image = placeholder_tk

# ---------------------------
# REPORT PANEL
# ---------------------------

report_frame = tk.Frame(root, bg="#10273f", width=520, height=520)
report_frame.grid(row=2, column=1, padx=20, pady=10)
report_frame.grid_propagate(False)

# Result Banner
result_label = tk.Label(report_frame,
                        text="Ready",
                        font=("Segoe UI", 20, "bold"),
                        bg="#1e293b",
                        fg="white",
                        width=25,
                        height=2)
result_label.pack(pady=15)

# Diagnosis
diagnosis_label = tk.Label(report_frame,
                           text="Diagnosis: --",
                           font=("Segoe UI", 16, "bold"),
                           bg="#10273f",
                           fg="white")
diagnosis_label.pack(pady=5)

# DR Status
dr_label = tk.Label(report_frame,
                    text="Diabetic Retinopathy: --",
                    font=("Segoe UI", 14),
                    bg="#10273f",
                    fg="white")
dr_label.pack(pady=5)

# Metrics Box
metrics = tk.Label(report_frame,
                   text="Waiting for analysis...",
                   font=("Consolas", 11),
                   bg="#020617",
                   fg="#22c55e",
                   width=40,
                   height=10,
                   justify="left",
                   anchor="nw",
                   padx=10,
                   pady=8)
metrics.pack(pady=15)

# ---------------------------
# Upload
# ---------------------------

def upload_image():
    global img_path, img_display

    file = filedialog.askopenfilename(
        filetypes=[("Retinal Images",
                    "*.png *.jpg *.jpeg *.tif *.ppm *.bmp")]
    )
    if not file:
        return

    img_path = file

    img = cv2.imread(file)
    if img is None:
        messagebox.showerror("Error", "Cannot read image.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (580, 480))
    img = Image.fromarray(img)
    img_display = ImageTk.PhotoImage(img)

    image_label.config(image=img_display)
    image_label.image = img_display

    result_label.config(text="Image Loaded", bg="#1e293b")
    diagnosis_label.config(text="Diagnosis: --")
    dr_label.config(text="Diabetic Retinopathy: --")
    metrics.config(text="Click 'Run Analysis' to begin.")

# ---------------------------
# Analysis
# ---------------------------

def analyze_image():
    global img_display

    if img_path is None:
        messagebox.showerror("Error", "Upload image first")
        return

    img      = cv2.imread(img_path)
    original = img.copy()

    green    = img[:, :, 1]
    clahe    = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(green)
    blur     = cv2.GaussianBlur(enhanced, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)

    vessel = cv2.adaptiveThreshold(
        tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, -2
    )
    vessel = cv2.medianBlur(vessel, 5)
    vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))

    vessel_pixels  = int(np.sum(vessel == 255))
    vessel_density = vessel_pixels / vessel.size
    vessel_length  = vessel_pixels

    # ── OPTIC DISC DETECTION (new) ──────────────
    blurred_disc = cv2.GaussianBlur(blur, (15, 15), 0)
    circles = cv2.HoughCircles(
        blurred_disc, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=100,
        param1=50, param2=30,
        minRadius=20, maxRadius=100
    )

    disc_cx, disc_cy, disc_r, disc_area = -1, -1, 0, 0
    if circles is not None:
        circles  = np.uint16(np.around(circles))
        disc_cx  = int(circles[0][0][0])
        disc_cy  = int(circles[0][0][1])
        disc_r   = int(circles[0][0][2])
        disc_area = int(np.pi * disc_r * disc_r)

    # ── MODEL PREDICTION (new) ──────────────────
    confidence = 0.0
    if model_loaded:
        feat       = np.array([[disc_cx, disc_cy, disc_r,
                                 disc_area, vessel_pixels,
                                 vessel_density, vessel_length]])
        ml_pred    = model.predict(feat)[0]
        probs      = model.predict_proba(feat)[0]
        confidence = round(max(probs) * 100, 1)
    else:
        ml_pred = 0

    # ── OVERLAY ─────────────────────────────────
    overlay        = cv2.resize(original, (580, 480))
    vessel_resized = cv2.resize(vessel,   (580, 480))

    vessel_color            = cv2.cvtColor(vessel_resized,
                                           cv2.COLOR_GRAY2BGR)
    vessel_color[:, :, 0]   = 0
    vessel_color[:, :, 1]   = vessel_resized
    vessel_color[:, :, 2]   = 0

    overlay = cv2.addWeighted(overlay, 0.85, vessel_color, 0.4, 0)

    # ── THRESHOLD LOGIC (unchanged from old code) ──

    if vessel_density < 0.025:
        risk       = "LOW VESSEL DENSITY"
        diagnosis  = "ABNORMAL (LOW DENSITY)"
        color      = "#3b82f6"
        dr         = "Unlikely"
        mark_color = (255, 100, 100)

    elif 0.025 <= vessel_density <= 0.032:
        risk       = "NORMAL"
        diagnosis  = "NORMAL"
        color      = "#22c55e"
        dr         = "No indication"
        mark_color = None

    else:
        risk       = "HIGH VESSEL DENSITY"
        diagnosis  = "ABNORMAL (HIGH DENSITY)"
        color      = "#ef4444"
        dr         = "Possible (Refer specialist)"
        mark_color = (0, 0, 255)

    # ── MARK ABNORMAL REGIONS (unchanged from old code) ──

    if mark_color is not None:
        contours, _ = cv2.findContours(vessel_resized,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > 5:
                    cv2.circle(overlay,
                               (int(x), int(y)),
                               int(radius) + 5,
                               mark_color, 2)
                    cv2.circle(overlay,
                               (int(x), int(y)),
                               int(radius) + 10,
                               mark_color, 1)

    # ── DRAW OPTIC DISC (new) ────────────────────
    if disc_r > 0:
        hr = 480 / original.shape[0]
        wr = 580 / original.shape[1]
        dx = int(disc_cx * wr)
        dy = int(disc_cy * hr)
        dr = int(disc_r  * min(hr, wr))
        cv2.circle(overlay, (dx, dy), dr,     (0, 255, 255), 2)
        cv2.circle(overlay, (dx, dy), dr + 4, (0, 180, 180), 1)
        cv2.putText(overlay, "Optic Disc",
                    (dx - 38, dy - dr - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

    # Convert and show
    overlay    = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay    = Image.fromarray(overlay)
    img_display = ImageTk.PhotoImage(overlay)
    image_label.config(image=img_display)
    image_label.image = img_display

    # ── UI UPDATE ────────────────────────────────

    result_label.config(text=risk, bg=color)
    diagnosis_label.config(text=f"Diagnosis: {diagnosis}")
    dr_label.config(text=f"Diabetic Retinopathy: {dr}")

    metrics.config(
        text=(
            f" Vessel Density  : {vessel_density:.4f}\n"
            f" Vessel Pixels   : {vessel_pixels:,}\n"
            f" Vessel Length   : {vessel_length:,}\n"
            f" ─────────────────────────────\n"
            f" Disc Radius     : {disc_r} px\n"
            f" Disc Area       : {disc_area} px²\n"
            f" ─────────────────────────────\n"
            f" RF Prediction   : {'NORMAL' if ml_pred == 0 else 'ABNORMAL'}\n"
            f" RF Confidence   : {confidence}%\n"
            f" Dataset         : DRIVE + STARE"
        )
    )

# ---------------------------
# Buttons
# ---------------------------

btn_frame = tk.Frame(root, bg="#0b1a2a")
btn_frame.grid(row=3, column=0, columnspan=2, pady=20)

tk.Button(btn_frame,
          text="Upload Image",
          command=upload_image,
          width=18,
          bg="#2563eb",
          fg="white",
          font=("Segoe UI", 12)).grid(row=0, column=0, padx=20)

tk.Button(btn_frame,
          text="Run Analysis",
          command=analyze_image,
          width=18,
          bg="#16a34a",
          fg="white",
          font=("Segoe UI", 12)).grid(row=0, column=1, padx=20)

root.mainloop()
