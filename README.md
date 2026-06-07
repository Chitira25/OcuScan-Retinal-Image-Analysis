OcuScan: A Classical Image Processing Framework for Retinal Vessel and Optic Disc Analysis
Overview
OcuScan is an automated retinal image analysis framework developed for retinal vessel segmentation, optic disc detection, biomarker extraction, and retinal disease screening using classical image processing and machine learning techniques.
The system processes retinal fundus images through a complete end-to-end pipeline and provides quantitative retinal biomarkers together with automated risk assessment through an interactive clinical dashboard.
Key Features
* Green channel based retinal image enhancement
* CLAHE contrast enhancement
* Retinal vessel segmentation using morphological image processing
* Optic disc detection and localization
* Retinal biomarker extraction
* XGBoost-based classification
* Interactive clinical dashboard
* Lightweight CPU-based deployment
* Interpretable and explainable outputs
Pipeline
Stage 1: Image Preprocessing
* Green Channel Extraction
* CLAHE Enhancement
* Gaussian Filtering
Stage 2: Vessel Segmentation
* Morphological Top-Hat Filtering
* Adaptive Thresholding
* Morphological Refinement
Stage 3: Optic Disc Detection
* Disc center localization
* Anatomical landmark identification
Stage 4: Feature Extraction
* Vessel Density
* Vessel Pixel Count
* Vessel Length
* Disc Radius
* Disc Area
Stage 5: Classification
* XGBoost Machine Learning Classifier
* Normal / Abnormal Screening
Technologies Used
* Python
* OpenCV
* NumPy
* Pandas
* Scikit-Learn
* XGBoost
* Matplotlib
* Tkinter
Dataset
The framework was developed and evaluated using retinal fundus image datasets:
* DRIVE Dataset
* STARE Dataset
Datasets are not included in this repository.
Project Structure
classification/ ? Model training and classification
preprocessing/ ? Image enhancement and preprocessing
segmentation/ ? Vessel and optic disc segmentation
features/ ? Biomarker extraction
visualization/ ? Clinical dashboard
utils/ ? Utility scripts and pipeline execution
How to Run
Install dependencies:
pip install -r requirements.txt
Run the pipeline:
python run_stare_pipeline.py
Results
* Automated retinal vessel extraction
* Optic disc localization
* Biomarker computation
* Clinical risk prediction
* Interactive visualization dashboard
Future Improvements
* Deep learning integration
* Multi-disease classification
* Web deployment
* Mobile deployment
* EHR integration
Authors
Chitira B
Deepasri M
Mathumithaa S
Department of Biomedical Engineering

