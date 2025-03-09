import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("cleaned_sheets/Sheet3_cleaned.xlsx")

# Ensure column names are case-insensitive
df.columns = df.columns.str.upper()

# Convert "APPROVAL" to numerical values
df["APPROVAL"] = df["APPROVAL"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

# Define possible categories
possible_categories = ['SC', 'ST', 'OBC', 'GENERAL', 'BPL', 'OEC']

# Ensure the 'CATEGORY' column is valid and contains only predefined categories
df['CATEGORY'] = df['CATEGORY'].apply(lambda x: x if x in possible_categories else 'GENERAL')  # Default to 'GENERAL' for unknown categories

# Encode "CATEGORY" column
label_encoder = LabelEncoder()
label_encoder.fit(possible_categories)  # Fit the encoder on all possible categories
df["CATEGORY"] = label_encoder.transform(df["CATEGORY"])

# Save the category encoder for future use
joblib.dump(label_encoder, "category_encoder.pkl")

# Define features
features = ["SCORE_INCOME", "SCORE_ACAD", "SCORE_DIST", "TOTAL SCORE", "CATEGORY"]

X = df[features]
y = df["APPROVAL"]
print(X)
# Check if a previous model exists
if os.path.exists("hostel_admission_model.pkl"):
    print("🔄 Previous model found. Loading...")
    model = joblib.load("hostel_admission_model.pkl")
    model.set_params(warm_start=True, n_estimators=model.n_estimators + 10)  # Add new estimators
else:
    print("🆕 No previous model found. Training from scratch.")
    model = RandomForestClassifier(n_estimators=500, random_state=510, warm_start=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# Train model (continue training if model exists)
model.fit(X_train, y_train)

# Save updated model
joblib.dump(model, "hostel_admission_model.pkl")

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Updated Model Accuracy: {accuracy:.2%}")