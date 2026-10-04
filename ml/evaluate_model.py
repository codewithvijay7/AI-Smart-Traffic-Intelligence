# ============================================================
# AI SMART TRAFFIC INTELLIGENCE
# RANDOM FOREST MODEL EVALUATION
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "traffic_data.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("AI SMART TRAFFIC INTELLIGENCE")
print("RANDOM FOREST MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print("Dataset loaded successfully.")

print()
print("Dataset shape:")
print(df.shape)

print()
print("Dataset columns:")

for index, column in enumerate(df.columns):
    print(
        f"{index + 1}. {column}"
    )


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "vehicle_count",
    "average_speed",
    "road_occupancy",
    "rainfall",
    "temperature",
    "hour",
    "day_of_week"
]


# ============================================================
# CHECK FEATURES
# ============================================================

print()
print("=" * 70)
print("CHECKING FEATURES")
print("=" * 70)

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print()
    print("ERROR: Missing feature columns:")

    for feature in missing_features:
        print(
            " -",
            feature
        )

    print()
    print("Available columns:")
    print(
        list(df.columns)
    )

    raise SystemExit(1)


print("All seven ML features found successfully.")


# ============================================================
# DETERMINE TARGET COLUMN
# ============================================================

# The CSV contains the seven input features followed
# by the traffic-class label.
#
# Therefore, use the final column as the target.
#
# This also prevents problems if the target column has
# a different name such as congestion, traffic, label, etc.

TARGET = df.columns[-1]


print()
print("=" * 70)
print("TARGET COLUMN")
print("=" * 70)

print(
    "Target column:",
    TARGET
)


# ============================================================
# SHOW TRAFFIC CLASSES
# ============================================================

print()
print("=" * 70)
print("TRAFFIC CLASS DISTRIBUTION")
print("=" * 70)

print(
    df[TARGET].value_counts()
)


# ============================================================
# PREPARE DATA
# ============================================================

data = df[
    FEATURES + [TARGET]
].copy()


# Remove rows containing missing values

data = data.dropna()


print()
print(
    "Rows after cleaning:",
    len(data)
)


# ============================================================
# INPUT AND OUTPUT
# ============================================================

X = data[
    FEATURES
]

y = data[
    TARGET
]


# Convert target values to strings

y = y.astype(str).str.strip()


# ============================================================
# SHOW CLASSES
# ============================================================

print()
print("Traffic classes found:")

for class_name in sorted(
    y.unique()
):

    count = (
        y == class_name
    ).sum()

    print(
        f" - {class_name}: {count}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Random Forest training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Generating predictions...")

y_pred = model.predict(
    X_test
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(

    y_test,
    y_pred,

    average="weighted",

    zero_division=0
)


recall = recall_score(

    y_test,
    y_pred,

    average="weighted",

    zero_division=0
)


f1 = f1_score(

    y_test,
    y_pred,

    average="weighted",

    zero_division=0
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(

        y_test,

        y_pred,

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


labels = sorted(
    y.unique()
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=labels
)


print()
print("Labels:")

print(
    labels
)


print()
print("Confusion Matrix:")

print(
    cm
)


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=labels
)


disp.plot(
    xticks_rotation=45
)


plt.title(
    "Random Forest - Traffic Level Confusion Matrix"
)


plt.tight_layout()


plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


importance_df = pd.DataFrame({

    "Feature":
        FEATURES,

    "Importance":
        model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print()

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(

    importance_df["Feature"],

    importance_df["Importance"]

)


plt.title(
    "Random Forest Feature Importance"
)


plt.xlabel(
    "Traffic Features"
)


plt.ylabel(
    "Importance"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.tight_layout()


plt.show()


# ============================================================
# MODEL PERFORMANCE GRAPH
# ============================================================

metric_names = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score"

]


metric_values = [

    accuracy * 100,

    precision * 100,

    recall * 100,

    f1 * 100

]


plt.figure(
    figsize=(9, 6)
)


plt.bar(

    metric_names,

    metric_values

)


plt.title(
    "Random Forest Model Performance"
)


plt.xlabel(
    "Evaluation Metric"
)


plt.ylabel(
    "Score (%)"
)


plt.ylim(
    0,
    100
)


plt.tight_layout()


plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


print()
print("Top 5 important features:")

for _, row in (
    importance_df.head(5).iterrows()
):

    print(

        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"

    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)