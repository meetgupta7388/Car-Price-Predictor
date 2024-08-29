from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

try:
    with open('LinearRegressionModel.pkl', 'rb') as file:
        model = pickle.load(file)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None  


try:
    car = pd.read_csv("Cleaned_Car_data.csv")
except Exception as e:
    print(f"Error loading car data: {e}")
    car = pd.DataFrame()  

@app.route('/')
def index():
    if car.empty:
        return "Car data not available"

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    companies.insert(0, "Select Company")
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fueltypes=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not available"

    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kilo_driven'))
        print(company, car_model, year, fuel_type, kms_driven)

        prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
        return str(np.round(prediction[0], 2))
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
