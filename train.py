import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Separate features and target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Function to predict diabetes
def predict_diabetes():
    print("Enter your details:")
    input_data = []
    for col in X.columns:
        value = float(input(f"{col}: "))
        input_data.append(value)
    
    # Load saved model and scaler
    loaded_model = joblib.load("diabetes_model.pkl")
    loaded_scaler = joblib.load("scaler.pkl")
    
    # Preprocess input data
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = loaded_scaler.transform(input_array)
    
    # Make prediction
    prediction = loaded_model.predict(input_scaled)
    
    if prediction[0] == 1:
        print("You might have diabetes. Please consult a doctor.")
    else:
        print("You are not likely to have diabetes.")

# Uncomment to allow user input
# predict_diabetes()
