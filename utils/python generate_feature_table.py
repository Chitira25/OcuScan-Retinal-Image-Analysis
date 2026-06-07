import pandas as pd

df = pd.read_csv("results/retina_dataset.csv")

# -----------------------
# Dataset composition
# -----------------------

counts = df['label'].value_counts()

composition = pd.DataFrame({
    "Class":["Normal","Abnormal","Total"],
    "Images":[counts.get(0,0),counts.get(1,0),len(df)]
})

# -----------------------
# Feature differentiation
# -----------------------

normal = df[df['label']==0]
abnormal = df[df['label']==1]

summary = pd.DataFrame({

"Feature":[
"Vessel Density",
"Vessel Pixels",
"Vessel Length"
],

"Normal Min":[
normal['vessel_density'].min(),
normal['vessel_pixels'].min(),
normal['vessel_length'].min()
],

"Normal Max":[
normal['vessel_density'].max(),
normal['vessel_pixels'].max(),
normal['vessel_length'].max()
],

"Abnormal Min":[
abnormal['vessel_density'].min(),
abnormal['vessel_pixels'].min(),
abnormal['vessel_length'].min()
],

"Abnormal Max":[
abnormal['vessel_density'].max(),
abnormal['vessel_pixels'].max(),
abnormal['vessel_length'].max()
]

})

# -----------------------
# Save Excel
# -----------------------

with pd.ExcelWriter("results/feature_differentiation_table.xlsx") as writer:
    
    composition.to_excel(writer,sheet_name="Dataset_Composition",index=False)
    summary.to_excel(writer,sheet_name="Feature_Differentiation",index=False)

print("Excel generated successfully")
