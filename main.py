from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[col]) for col in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]]
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        result = "You might have diabetes. Please consult a doctor." if prediction[0] == 1 else "You are not likely to have diabetes."
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)