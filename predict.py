import pandas as pd
import joblib

# Load model and encoder
model_path = "hostel_admission_model.pkl"
encoder_path = "category_encoder.pkl"

print("🔄 Loading trained model and category encoder...")
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Function to preprocess input data
def preprocess_input(score_income, score_acad, score_dist, category):
    total_score = score_income + score_acad + score_dist  # Calculate total score
    
    # Ensure category is in the known classes
    if category in label_encoder.classes_:
        category_encoded = label_encoder.transform([category])[0]
    else:
        print(f"⚠️ Warning: Unseen category '{category}'. Assigning default encoding.")
        category_encoded = -1  # Assign a special value for unseen categories

    # Print debug information
    print(f"\n[DEBUG] Processed Input Data:")
    print(f"SCORE_INCOME: {score_income}, SCORE_ACAD: {score_acad}, SCORE_DIST: {score_dist}, TOTAL SCORE: {total_score}, CATEGORY: {category_encoded}\n")
    
    # Create a DataFrame to match model input structure
    input_data = pd.DataFrame([[score_income, score_acad, score_dist, total_score, category_encoded]], 
                              columns=["SCORE_INCOME", "SCORE_ACAD", "SCORE_DIST", "TOTAL SCORE", "CATEGORY"])
    return input_data, total_score

# Function to predict hostel admission
def predict_hostel_admission(score_income, score_acad, score_dist, category):
    X_input, total_score = preprocess_input(score_income, score_acad, score_dist, category)
    
    try:
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0][1]  # Probability of approval (class 1)
        admission_probability = prediction_proba * 100  # Convert to percentage
        return prediction, admission_probability, total_score
    except ValueError as e:
        print(f"⚠️ Error during prediction: {e}")
        return None, None, None

# Take input from console
if __name__ == "__main__":
    print("\n🎓 Hostel Admission Predictor")
    
    try:
        score_income = float(input("Enter Score (Income): "))
        score_acad = float(input("Enter Score (Academic): "))
        score_dist = float(input("Enter Score (District): "))
        category = input("Enter Category (e.g., GEN, OBC, SC, ST, BPL): ").strip().upper()

        prediction, admission_probability, total_score = predict_hostel_admission(score_income, score_acad, score_dist, category)

        if prediction is not None:
            result = "✅ Approved" if prediction == 1 else "❌ Not Approved"
            print(f"\n🎯 Total Score: {total_score}")
            print(f"🎯 Prediction: {result}")
            print(f"🎯 Probability of Admission: {admission_probability:.2f}%")
        
    except ValueError as e:
        print(f"⚠️ Invalid input! Please enter numerical values for scores. {e}")
