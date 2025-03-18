# Diabetes Prediction System

## Overview
This project is a **Diabetes Prediction System** built using **Flask, Scikit-Learn, Pandas, and NumPy**. The system uses a **Random Forest Classifier** trained on the **Pima Indians Diabetes Dataset** to predict whether a person has diabetes based on input medical parameters.

## Features
- Machine Learning model trained using **Random Forest Classifier**.
- **Flask Web Application** for user interaction.
- **StandardScaler** is used for data preprocessing.
- Model and scaler are saved using **joblib** for future predictions.
- **HTML-based UI** to input user details and display predictions.

## Dataset
The dataset used is **diabetes.csv**, which includes the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target: 0 = No Diabetes, 1 = Diabetes)

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:

```sh
pip install flask pandas numpy scikit-learn joblib
```

## Running the Application
1. Clone the repository:
```sh
git clone https://github.com/krishkrishna03/Diabetes_Prediction_System.git
cd diabetes-prediction-system
```

2. Run the Flask application:
```sh
python app.py
```

3. Open your browser and visit:
```
http://127.0.0.1:5000/
```

## Project Structure
```
├── diabetes.csv                # Dataset
├── app.py                      # Flask Application
├── diabetes_model.pkl          # Saved Machine Learning Model
├── scaler.pkl                  # Saved Scaler
├── templates/
│   ├── index.html              # Frontend UI
├── static/
│   ├── style.css               # CSS Styles (if any)
└── README.md                   # Project Documentation
```

## Usage
1. Enter the required details in the web form.
2. Click on **Predict**.
3. The system will analyze and display the diabetes prediction result.

## License
This project is open-source and available for modification and distribution.
