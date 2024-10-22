from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the pre-trained model
with open('best_knn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    age = float(request.form['age'])
    gender = float(request.form['gender'])
    experience_years = float(request.form['experience_years'])
    job_role = float(request.form['job_role'])
    job_satisfaction = float(request.form['job_satisfaction'])
    performance_rating = float(request.form['performance_rating'])
    training_hours = float(request.form['training_hours'])
    salary = float(request.form['salary'])

    # Prepare the input array in the correct order
    input_features = np.array([[age, gender, experience_years, job_role, job_satisfaction,
                                performance_rating, training_hours, salary]])

    # Use the model to predict
    prediction = model.predict(input_features)
    
    # Interpret the result (assuming 1 means attrition and 0 means no attrition)
    output = 'Yes' if prediction[0] == 1 else 'No'

    # Return the result to the result.html template
    return render_template('result.html', prediction_text=f'Employee Attrition: {output}')

if __name__ == '__main__':
    app.run(debug=True)
