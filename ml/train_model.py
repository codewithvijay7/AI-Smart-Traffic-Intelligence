import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


DATASET_PATH = os.path.join(
    DATASET_DIR,
    "traffic_data.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "traffic_model.pkl"
)


# ==========================================================
# 1. GENERATE DEVELOPMENT TRAFFIC DATASET
# ==========================================================

print("\n==========================================")
print("AI SMART TRAFFIC INTELLIGENCE")
print("==========================================")

print("\nGenerating traffic dataset...")


np.random.seed(42)

NUMBER_OF_RECORDS = 5000


# ----------------------------------------------------------
# Traffic features
# ----------------------------------------------------------

vehicle_count = np.random.randint(
    20,
    500,
    NUMBER_OF_RECORDS
)

average_speed = np.random.randint(
    10,
    90,
    NUMBER_OF_RECORDS
)

road_occupancy = np.random.uniform(
    5,
    95,
    NUMBER_OF_RECORDS
)

rainfall = np.random.uniform(
    0,
    80,
    NUMBER_OF_RECORDS
)

temperature = np.random.uniform(
    15,
    45,
    NUMBER_OF_RECORDS
)

hour = np.random.randint(
    0,
    24,
    NUMBER_OF_RECORDS
)

day_of_week = np.random.randint(
    0,
    7,
    NUMBER_OF_RECORDS
)


# ==========================================================
# TRAFFIC CONGESTION SCORE
# ==========================================================

traffic_score = (

    vehicle_count * 0.35

    + road_occupancy * 0.30

    - average_speed * 0.25

    + rainfall * 0.05

    + temperature * 0.02

)


# Rush-hour effect

rush_hour = (
    ((hour >= 7) & (hour <= 10))
    |
    ((hour >= 17) & (hour <= 21))
)

traffic_score += rush_hour * 20


# Weekend reduction

weekend = (
    (day_of_week == 5)
    |
    (day_of_week == 6)
)

traffic_score -= weekend * 8


# Add small random variation

traffic_score += np.random.normal(
    0,
    8,
    NUMBER_OF_RECORDS
)


# ==========================================================
# CONGESTION LABEL
# ==========================================================

def get_congestion_level(score):

    if score < 100:
        return "Low"

    elif score < 160:
        return "Moderate"

    elif score < 220:
        return "High"

    else:
        return "Severe"


congestion_level = [

    get_congestion_level(score)

    for score in traffic_score

]


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame({

    "vehicle_count":
        vehicle_count,

    "average_speed":
        average_speed,

    "road_occupancy":
        road_occupancy,

    "rainfall":
        rainfall,

    "temperature":
        temperature,

    "hour":
        hour,

    "day_of_week":
        day_of_week,

    "congestion_level":
        congestion_level

})


# ==========================================================
# SAVE DATASET
# ==========================================================

df.to_csv(
    DATASET_PATH,
    index=False
)


print(
    "\nDataset created successfully."
)

print(
    "Dataset location:"
)

print(
    DATASET_PATH
)


print(
    "\nDataset shape:",
    df.shape
)


print(
    "\nFirst 5 records:"
)

print(
    df.head()
)


# ==========================================================
# 2. PREPARE DATA FOR MACHINE LEARNING
# ==========================================================

print("\n==========================================")
print("PREPARING DATA")
print("==========================================")


FEATURES = [

    "vehicle_count",

    "average_speed",

    "road_occupancy",

    "rainfall",

    "temperature",

    "hour",

    "day_of_week"

]


TARGET = "congestion_level"


X = df[FEATURES]

y = df[TARGET]


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ==========================================================
# 3. RANDOM FOREST MODEL
# ==========================================================

print("\n==========================================")
print("TRAINING RANDOM FOREST")
print("==========================================")


model = RandomForestClassifier(

    n_estimators=200,

    max_depth=15,

    min_samples_split=4,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


print(
    "\nModel training completed."
)


# ==========================================================
# 4. PREDICTION
# ==========================================================

y_prediction = model.predict(
    X_test
)


# ==========================================================
# 5. MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)


print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")


print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_prediction
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_prediction
    )
)


# ==========================================================
# 6. FEATURE IMPORTANCE
# ==========================================================

print("\n==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")


importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(

    by="importance",

    ascending=False

)


print(
    importance.to_string(
        index=False
    )
)


# ==========================================================
# 7. SAVE MODEL
# ==========================================================

joblib.dump(

    {

        "model": model,

        "features": FEATURES

    },

    MODEL_PATH

)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")


print(
    "\nModel location:"
)

print(
    MODEL_PATH
)


print(
    "\nAI traffic prediction model is ready!"
)


# ==========================================================
# 8. TEST WITH SAMPLE TRAFFIC
# ==========================================================

print("\n==========================================")
print("SAMPLE PREDICTION")
print("==========================================")


sample = pd.DataFrame({

    "vehicle_count": [350],

    "average_speed": [25],

    "road_occupancy": [80],

    "rainfall": [15],

    "temperature": [32],

    "hour": [18],

    "day_of_week": [2]

})


sample_prediction = model.predict(
    sample
)


sample_probability = model.predict_proba(
    sample
)


print(
    "\nSample Traffic:"
)

print(
    sample.to_string(
        index=False
    )
)


print(
    "\nPredicted Congestion:"
)

print(
    sample_prediction[0]
)


print(
    "\nPrediction Probabilities:"
)

classes = model.classes_

for class_name, probability in zip(
    classes,
    sample_probability[0]
):

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )


print("\n==========================================")
print("TRAINING FINISHED")
print("==========================================")