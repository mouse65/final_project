from flask import Flask, request, jsonify,render_template
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib

app = Flask(__name__)

# Загружаем модель один раз при запуске сервера
##model = CatBoostRegressor()
##model.load_model("catboost_rf_model.cbm")
model = joblib.load("stack_model.pkl")


# Маршрут по умолчанию
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    beds = int(request.form['beds'])
    baths = int(request.form['baths'])
    has_pool = int(request.form['has_pool'])
    fireplace_count = int(request.form['fireplace_count'])
#
    # Пример: создаём DataFrame с нужными признаками
    state_value = float(request.form['state'])
    stories_clean = float(request.form['stories'])

    input_data = pd.DataFrame([{
        "baths": baths,
        "sqft": sqft,
        "beds": beds,
        "state": state_value,
        "has_pool": has_pool,
        "fireplace_count": fireplace_count,
        "stories_clean": stories_clean,
        "home_Year_built": 2019.,
        "lotsize_sqft": 7318.,
        "zipcode_prefix": 256,
        "city_grouped": 33,
        "age_since_remodel": 6,
        "parking_group": 12.74,
        "school_rating_mean": 8.8,
        "school_min_dist": 1.5,
        "zipcode_encod_log": 13.96,
        "city_encod_log": 13.16,
    
    # One-hot encoded property types (only one should be 1)
        "propertyType_condo": 0,
        "propertyType_coop": 0,
        "propertyType_land": 0,
        "propertyType_luxury": 0,
        "propertyType_manufactured": 0,
        "propertyType_multi-family": 0,
        "propertyType_other": 0,
        "propertyType_single family": 1,
        "propertyType_townhouse": 0,

    # Status flags (only one should be 1)
        "status_clean_coming soon": 0,
        "status_clean_contingent": 0,
        "status_clean_for sale": 1,
        "status_clean_foreclosure": 0,
        "status_clean_other": 0,
        "status_clean_pending": 0,
        "status_clean_rental": 0,
        "status_clean_sold": 0,
        "status_clean_under contract": 0,

    # Heating types (only one should be 1)
        "heating_group_central": 1,
        "heating_group_electric": 0,
        "heating_group_forced_air": 0,
        "heating_group_gas": 0,
        "heating_group_heat_pump": 0,
        "heating_group_other": 0,
        "heating_group_radiant": 0,
        "heating_group_wall": 0,

    # Cooling types (only one should be 1)
        "cooling_group_electric": 1,
        "cooling_group_evaporative": 0,
        "cooling_group_gas": 0,
        "cooling_group_heat_pump": 0,
        "cooling_group_none": 0,
        "cooling_group_other": 0,
        "cooling_group_wall_window": 0
}])


    log_price = model.predict(input_data)[0]  # получаем логарифм цены
    price = np.expm1(log_price)               # преобразуем обратно в доллары

    formatted_price = f"${price:,.1f}"        # красиво форматируем

    return f"Оценочная стоимость недвижимости: {formatted_price}"


# Запуск сервера
if __name__ == '__main__':
    app.run(debug=True, port=5050)
