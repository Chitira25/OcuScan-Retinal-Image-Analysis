import pandas as pd

df = pd.read_csv("results/disc_features.csv")

print("Min radius:", df["disc_radius_px"].min())
print("Max radius:", df["disc_radius_px"].max())
print("Mean radius:", df["disc_radius_px"].mean())
