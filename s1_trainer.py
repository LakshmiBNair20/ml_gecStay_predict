import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load dataset from CSV
df = pd.read_csv("lh.csv")

# Ensure column names are case-insensitive
df.columns = df.columns.str.upper()

# Add the "ADMITTED" column and set all existing rows to 1 (admitted)
df["ADMITTED"] = 1

# Synthetic data for non-admitted students
synthetic_data = {
    "Sl. No.": list(range(75, 135)),  # 60 synthetic students
    "Name": [f"Student{i}" for i in range(75, 135)],
    "Class": [
        "CS", "ECE", "ME", "EEE", "CEE", "CS", "ECE", "ME", "EEE", "CEE",
        "CS", "ECE", "ME", "EEE", "CEE", "CS", "ECE", "ME", "EEE", "CEE",
        "CS", "ECE", "ME", "EEE", "CEE", "CS", "ECE", "ME", "EEE", "CEE",
        "CS", "ECE", "ME", "EEE", "CEE", "CS", "ECE", "ME", "EEE", "CEE",
        "CS", "ECE", "ME", "EEE", "CEE", "CS", "ECE", "ME", "EEE", "CEE",
        "CS", "ECE", "ME", "EEE", "CEE", "CS", "ECE", "ME", "EEE", "CEE"
    ],
    "Category": [
        "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL",
        "BPL", "BPL", "BPL", "BPL", "BPL", "BPL", "BPL", "BPL", "BPL", "BPL",
        "SC", "SC", "SC", "SC", "SC", "SC", "SC", "SC", "SC", "SC",
        "ST", "ST", "ST", "ST", "ST", "GENERAL", "GENERAL", "GENERAL", "GENERAL", "GENERAL",
        "BPL", "BPL", "BPL", "BPL", "BPL", "SC", "SC", "SC", "SC", "SC",
        "ST", "ST", "ST", "GENERAL", "GENERAL", "BPL", "BPL", "SC", "SC", "ST"
    ],
    "Score Income": [
        # GENERAL category (below 54.2)
        53.1, 52.8, 51.5, 50.2, 49.7, 48.3, 47.9, 46.5, 45.2, 44.8, 
        # BPL category (below 41.7)
        40.9, 40.2, 39.5, 38.9, 38.2, 37.8, 36.9, 36.1, 35.8, 35.2,
        # SC category (below 41.7)
        40.8, 40.5, 39.8, 39.2, 38.5, 38.1, 37.5, 37.0, 36.4, 35.9,
        # ST category (below 69.1)
        68.5, 68.0, 67.5, 67.0, 66.5, 
        # Mix of categories with borderline scores
        54.0, 53.9, 53.8, 53.7, 54.1,
        41.6, 41.5, 41.4, 41.3, 41.2, 
        41.6, 41.5, 41.4, 41.3, 41.2,
        69.0, 68.9, 68.8, 
        # Additional GENERAL and BPL with very low scores
        42.0, 43.0, 
        35.0, 37.0, 34.0, 36.0, 33.0
    ],
    "Score Dist.": [
        # GENERAL category (below 11.25)
        11.0, 10.8, 10.5, 10.2, 9.8, 9.5, 9.2, 8.8, 8.5, 8.2,
        # BPL category (below 5.0)
        4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0,
        # SC category (below 6.25)
        6.0, 5.8, 5.6, 5.4, 5.2, 5.0, 4.8, 4.6, 4.4, 4.2,
        # ST category (below 0.625)
        0.6, 0.55, 0.5, 0.45, 0.4,
        # Mix of categories with borderline scores
        11.2, 11.1, 11.0, 10.9, 10.8,
        4.9, 4.8, 4.7, 4.6, 4.5,
        6.2, 6.1, 6.0, 5.9, 5.8,
        0.62, 0.61, 0.60,
        # Additional with low scores
        7.0, 8.0,
        2.5, 2.7, 2.0, 1.8, 1.5
    ],
    "Total Score": [
        # GENERAL category (below 77.99)
        77.8, 77.5, 77.0, 76.5, 76.0, 75.5, 75.0, 74.5, 74.0, 73.5,
        # BPL category (below 72.76)
        72.5, 72.0, 71.5, 71.0, 70.5, 70.0, 69.5, 69.0, 68.5, 68.0,
        # SC category (below 52.99)
        52.8, 52.5, 52.0, 51.5, 51.0, 50.5, 50.0, 49.5, 49.0, 48.5,
        # ST category (below 69.81)
        69.7, 69.5, 69.3, 69.1, 68.9,
        # Mix of categories with borderline scores
        77.9, 77.8, 77.7, 77.6, 77.5,
        72.7, 72.6, 72.5, 72.4, 72.3,
        52.9, 52.8, 52.7, 52.6, 52.5,
        69.8, 69.7, 69.6,
        # Additional with lower scores
        65.0, 67.0,
        60.0, 62.0, 55.0, 57.0, 50.0
    ],
    "KEAM Rank": [
        # Distributed KEAM ranks from 3000 to 99000
        3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000,
        23000, 25000, 27000, 29000, 31000, 33000, 35000, 37000, 39000, 41000,
        43000, 45000, 47000, 49000, 51000, 53000, 55000, 57000, 59000, 61000,
        63000, 65000, 67000, 69000, 71000, 73000, 75000, 77000, 79000, 81000,
        83000, 85000, 87000, 89000, 91000, 93000, 95000, 97000, 99000, 98000,
        96000, 94000, 92000, 90000, 88000, 86000, 84000, 82000, 80000, 78000
    ],
    "ADMITTED": [0] * 60  # All are not admitted
}
# Convert synthetic data to a DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Make sure we have the same column name case consistency
synthetic_df.columns = synthetic_df.columns.str.upper()

# Combine the original data with synthetic data
df = pd.concat([df, synthetic_df], ignore_index=True)

# Check for NaN values and fill them if they exist
print(f"NaN values in ADMITTED column before cleaning: {df['ADMITTED'].isna().sum()}")
df["ADMITTED"] = df["ADMITTED"].fillna(0)  # Fill NaN values with 0 (not admitted)

# Handle categories manually by assigning all possible categories
possible_categories = ['GENERAL', 'BPL', 'SC', 'ST', 'OBC', 'OEC', 'BH']

# Ensure the 'CATEGORY' column is valid and contains only predefined categories
df['CATEGORY'] = df['CATEGORY'].apply(lambda x: x if x in possible_categories else 'GENERAL')  # Default to 'GENERAL' for unknown categories

# Fill potential NaN values in features
df["SCORE INCOME"] = df["SCORE INCOME"].fillna(df["SCORE INCOME"].median())
df["SCORE DIST."] = df["SCORE DIST."].fillna(df["SCORE DIST."].median())
df["TOTAL SCORE"] = df["TOTAL SCORE"].fillna(df["TOTAL SCORE"].median())
df["KEAM RANK"] = df["KEAM RANK"].fillna(df["KEAM RANK"].median())
df["CATEGORY"] = df["CATEGORY"].fillna("GENERAL")

# Calculate weighted score based on the specified weights
# 70% for income score, 30% for distance score
df["WEIGHTED_SCORE"] = (df["SCORE INCOME"] * 0.7) + (df["SCORE DIST."] * 0.3)

# Create category priority (higher number = higher priority)
category_priority = {
    'BPL': 5,   # Highest priority
    'SC': 4,
    'ST': 3,
    'OBC': 2,
    'OEC': 1,
    'BH': 1,
    'GENERAL': 0  # Lowest priority
}

# Map categories to their priority values
df["CATEGORY_PRIORITY"] = df["CATEGORY"].apply(lambda x: category_priority.get(x, 0) if isinstance(x, str) else 0)

# Encode "CATEGORY" column for the model
label_encoder = LabelEncoder()
df["CATEGORY_ENCODED"] = label_encoder.fit_transform(df["CATEGORY"])

# Save the category encoder for future use
joblib.dump(label_encoder, "category_encoder.pkl")

# Create a function to resolve ties using KEAM rank
def get_admission_priority(row):
    # First priority: Weighted score (higher is better)
    priority = row["WEIGHTED_SCORE"] * 1000
    
    # Second priority: Category priority (higher is better)
    priority += row["CATEGORY_PRIORITY"] * 100
    
    # Third priority (tiebreaker): KEAM rank (lower is better)
    # Convert rank to a small decimal addition so it doesn't override the main priorities
    max_rank = df["KEAM RANK"].max() + 1
    rank_priority = (max_rank - row["KEAM RANK"]) / max_rank
    priority += rank_priority
    
    return priority

# Calculate overall admission priority score


# Define features for the model
features = ["SCORE INCOME", "SCORE DIST.", "WEIGHTED_SCORE", "CATEGORY_ENCODED", "KEAM RANK"]
X = df[features]
y = df["ADMITTED"]

# Final check for NaN values before SMOTE
print("Final check for NaN values:")
print("X NaN counts:", X.isna().sum().sum())
print("y NaN counts:", y.isna().sum())

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=420)

# Create a pipeline with scaling and model
pipeline = make_pipeline(
    StandardScaler(),  # Scale numerical features
    RandomForestClassifier(n_estimators=200, random_state=420, max_depth=5)  # Limit tree depth to reduce overfitting
)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save updated model
joblib.dump(pipeline, "hostel_admission_model.pkl")

# Evaluate on the test set
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")

# Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
if hasattr(pipeline[-1], 'feature_importances_'):
    importances = pipeline[-1].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values(by='Importance', ascending=False))

# Create a function for making predictions for new data
def predict_admission_probability(data):
    """
    Predict admission probability for new student data.
    
    Parameters:
    data (dict or DataFrame): Student information with the following keys:
                             'SCORE INCOME', 'SCORE DIST.', 'CATEGORY', 'KEAM RANK'
    
    Returns:
    float: Probability of admission (0-1)
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Ensure all required columns exist
    required_cols = ['SCORE INCOME', 'SCORE DIST.', 'CATEGORY', 'KEAM RANK']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Apply the same transformations as in training
    data["WEIGHTED_SCORE"] = (data["SCORE INCOME"] * 0.7) + (data["SCORE DIST."] * 0.3)
    data["CATEGORY_PRIORITY"] = data["CATEGORY"].apply(lambda x: category_priority.get(x, 0) if isinstance(x, str) else 0)
    data["CATEGORY_ENCODED"] = label_encoder.transform(data["CATEGORY"])
    
    # Calculate admission priority
    max_rank = df["KEAM RANK"].max() + 1
    data["ADMISSION_PRIORITY"] = data.apply(lambda row: 
        (row["WEIGHTED_SCORE"] * 1000) + 
        (row["CATEGORY_PRIORITY"] * 100) + 
        ((max_rank - row["KEAM RANK"]) / max_rank), 
        axis=1)
    
    # Prepare features in the same order as training
    X_new = data[features]
    
    # Get probability predictions
    proba = pipeline.predict_proba(X_new)
    
    # Return probability of admission (class 1)
    return proba[:, 1]

# Save the prediction function
joblib.dump(predict_admission_probability, "predict_admission_function.pkl")

# Create a sample prediction code
sample_prediction_code = """
# Example of how to use the prediction function:
import joblib

# Load the prediction function
predict_admission = joblib.load("predict_admission_function.pkl")

# Sample new student data
new_student = {
    'SCORE INCOME': 70,  # Income score (higher = lower income = higher priority)
    'SCORE DIST.': 25,   # Distance score (higher = farther = higher priority)
    'CATEGORY': 'SC',    # Student category
    'KEAM RANK': 5000    # KEAM rank (lower = better)
}

# Get admission probability
probability = predict_admission(new_student)
print(f"Probability of admission: {probability[0]:.2%}")
"""

# Save the sample code
with open("sample_prediction_code.py", "w") as f:
    f.write(sample_prediction_code)

print("\nA sample prediction code has been saved to 'sample_prediction_code.py'")