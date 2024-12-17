from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and data
model = pickle.load(open('linear-regression_model.pkl', 'rb'))
car = pd.read_csv('cleaned_car_new_data.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models_by_company = car.groupby('company')['name'].apply(lambda x: sorted(set(x))).to_dict()
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index.html',
                           companies=companies,
                           car_models_by_company=car_models_by_company,
                           years=year,
                           fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Fetch form data
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Prepare data for prediction
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))

    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run()
