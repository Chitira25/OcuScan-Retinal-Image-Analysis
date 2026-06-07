import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier


# --------------------------------
# Load dataset
# --------------------------------

data = pd.read_csv("../results/retina_dataset.csv")

print("Dataset loaded")
print("Total samples:", len(data))


# --------------------------------
# Features and labels
# --------------------------------

X = data.drop(["id","label"], axis=1)
y = data["label"]


# --------------------------------
# Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


# --------------------------------
# XGBoost Model
# --------------------------------

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------
# Predictions
# --------------------------------

pred = model.predict(X_test)


# --------------------------------
# Evaluation
# --------------------------------

print("\nModel Performance")
print("-----------------------")

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))


# --------------------------------
# Feature Importance
# --------------------------------

print("\nFeature Importance")
print("-----------------------")

importance = model.feature_importances_

for name, score in zip(X.columns, importance):
    print(name, ":", round(score,4))
