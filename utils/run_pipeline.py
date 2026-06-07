import subprocess
import os

PROJECT_PATH = r"C:\Users\chiti\OneDrive\Documents\OcuScan project"

scripts = [
    os.path.join(PROJECT_PATH,"features","detect_optic_disc.py"),
    os.path.join(PROJECT_PATH,"segment_vessels.py"),
    os.path.join(PROJECT_PATH,"extract_vessel_features.py"),
    os.path.join(PROJECT_PATH,"build_dataset.py")
]

print("\nSTARTING FULL RETINAL ANALYSIS PIPELINE\n")

for script in scripts:

    print("\n-----------------------------------")
    print("Running:", os.path.basename(script))
    print("-----------------------------------")

    result = subprocess.run(
        ["python", script],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print("ERROR:", result.stderr)

print("\nPIPELINE FINISHED")
print("\nCheck outputs inside the results folder")
