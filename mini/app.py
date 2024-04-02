from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load("decision_tree_model.pkl")

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form

    # Create a DataFrame from the form data
    input_data = {
        'Age1': [float(data['Age'])],
        'BMI': [float(data['BMI'])],
        'fFN': [float(data['fFN'])],
        'Diabetes': [int(data['Diabetes'])],
        'Asthma': [int(data['Asthma'])],
        'Count Contraction': [int(data['Count Contraction'])],
        'lenght of contraction': [int(data['lenght of contraction'])],
        'chronic hypertension': [int(data['chronic hypertension'])],
        'hypertensive disorders of pregnancy': [int(data['hypertensive disorders of pregnancy'])]
    }
    input_df = pd.DataFrame(input_data)

    # Make prediction
    prediction = loaded_model.predict(input_df)

    # Return the prediction as JSON response
    if prediction[0] == 1:
        result = "Their is a risk of Preterm Birth."
    else:
        result = "It's a Normal Birth."
    
    return render_template('index1.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
