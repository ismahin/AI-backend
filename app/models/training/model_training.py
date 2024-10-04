# model_training.py

import pandas as pd
from prophet import Prophet
import joblib

# Function to convert YEAR and DOY to date
def convert_to_date(row):
    return pd.to_datetime(row['YEAR'], format='%Y') + pd.to_timedelta(row['DOY'] - 1, unit='D')

# Load the historical weather data
data = pd.read_csv('../../../weather.csv')

# Convert YEAR and DOY to date
data['date'] = data.apply(convert_to_date, axis=1)

# Rename columns to match expected names
data.rename(columns={
    'T2M': 'temperature',
    'PRECTOTCORR': 'rainfall',
    'RH2M': 'humidity'
}, inplace=True)

# Ensure data is sorted by date
data = data.sort_values(by='date')

# Handle missing values if any
data = data[['date', 'temperature', 'rainfall', 'humidity']].interpolate()

# Unit conversions if necessary
# Convert temperature from Kelvin to Celsius if T2M is in Kelvin
data['temperature'] = data['temperature'] - 273.15

# Convert rainfall from meters to millimeters if PRECTOTCORR is in meters
data['rainfall'] = data['rainfall'] * 1000

# Ensure humidity is within 0-100%
data['humidity'] = data['humidity'].clip(0, 100)

# Function to prepare data for Prophet
def prepare_prophet_data(df, column_name):
    prophet_df = df[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
    return prophet_df

# Train Prophet model for temperature
temp_df = prepare_prophet_data(data, 'temperature')
temp_model = Prophet()
temp_model.fit(temp_df)
joblib.dump(temp_model, 'temp_prophet_model.pkl')

# Train Prophet model for rainfall
rain_df = prepare_prophet_data(data, 'rainfall')
rain_model = Prophet()
rain_model.fit(rain_df)
joblib.dump(rain_model, 'rain_prophet_model.pkl')

# Train Prophet model for humidity
hum_df = prepare_prophet_data(data, 'humidity')
hum_model = Prophet()
hum_model.fit(hum_df)
joblib.dump(hum_model, 'hum_prophet_model.pkl')

print("Models trained and saved successfully.")
