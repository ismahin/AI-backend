from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import joblib
import json
import traceback
from pydantic import BaseModel
from typing import List

from app.models.disease_model import DiseaseModel
from app.utils.helpers import get_cure_recommendations, draw_red_square
from transformers import pipeline

router = APIRouter()

# =========================
# Initialize Models and Pipelines
# =========================

# Initialize the disease model
model = DiseaseModel()

# Initialize the chat pipeline
# chat_pipeline = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

# # Define the agriculture-focused prompt
# prompt_template = """
# You are an expert agricultural assistant. Your job is to provide helpful and concise answers to questions related to farming, crop management, livestock, and agricultural best practices. Keep your responses informative, clear, and specific to the topic of agriculture.

# Q: {question}
# A:"""

# Load crop data from seasonal JSON files
def load_crops():
    with open('app/data/summer_crops.json', 'r') as f:
        summer_crops = json.load(f)
    with open('app/data/rainy_crops.json', 'r') as f:
        rainy_crops = json.load(f)
    with open('app/data/winter_crops.json', 'r') as f:
        winter_crops = json.load(f)
    return {
        'summer': summer_crops,
        'rainy': rainy_crops,
        'winter': winter_crops
    }

crops_by_season = load_crops()

# Load trained Prophet models
temp_model = joblib.load('app/models/temp_prophet_model.pkl')
rain_model = joblib.load('app/models/rain_prophet_model.pkl')
hum_model = joblib.load('app/models/hum_prophet_model.pkl')

# =========================
# Define Data Models
# =========================

# For Crop Suggestion
class ForecastRequest(BaseModel):
    start_date: str  # 'YYYY-MM-DD'
    end_date: str    # 'YYYY-MM-DD'

class CropSuggestion(BaseModel):
    name: str
    # image_link: str  # Uncomment if you have image links
    season: str

# =========================
# Helper Functions
# =========================

# Function to map dates to seasons
def get_season(date):
    month = date.month
    if 4 <= month <= 6:
        return 'summer'
    elif 7 <= month <= 11:
        return 'rainy'
    else:
        return 'winter'

# Function to forecast weather parameters using Prophet
def forecast_weather(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)

    # Prepare future dataframe for Prophet
    future_dates = pd.DataFrame({'ds': date_range})

    # Forecast temperature
    temp_forecast = temp_model.predict(future_dates)
    temp_values = temp_forecast['yhat'].values

    # Forecast rainfall
    rain_forecast = rain_model.predict(future_dates)
    rain_values = rain_forecast['yhat'].values

    # Forecast humidity
    hum_forecast = hum_model.predict(future_dates)
    hum_values = hum_forecast['yhat'].values

    # Combine forecasts into a DataFrame
    forecast_df = pd.DataFrame({
        'date': date_range,
        'temperature': temp_values,
        'rainfall': rain_values,
        'humidity': hum_values
    })

    return forecast_df

# Function to aggregate forecast data by season
def aggregate_forecast_by_season(forecast_df):
    forecast_df['season'] = forecast_df['date'].apply(get_season)
    season_aggregates = forecast_df.groupby('season').agg({
        'temperature': 'mean',
        'rainfall': 'sum',
        'humidity': 'mean',
        'date': 'count'
    }).rename(columns={'date': 'days'})
    return season_aggregates

# Function to suggest crops based on aggregated data
def suggest_crops(season_aggregates):
    suggested_crops = []
    for season, data in season_aggregates.iterrows():
        season_crops = crops_by_season.get(season, [])
        matching_crops = []
        print(f"\nSeason: {season}")
        print(f"Aggregated Data:\n{data}\n")
        print(f"Available Crops for Season ({season}): {[crop['name'] for crop in season_crops]}")

        for crop in season_crops:
            temp_min, temp_max = crop['temperatureRange']
            rain_min, rain_max = crop['rainfallRange']
            hum_min, hum_max = crop['humidityRange']
            # If 'minimum_days' is included, uncomment the next line
            # min_days = crop.get('minimum_days', 0)
            days = data['days']

            # Debugging prints
            print(f"Evaluating Crop: {crop['name']}")
            print(f"Crop Ranges - Temp: {temp_min}-{temp_max}, Rain: {rain_min}-{rain_max}, Humidity: {hum_min}-{hum_max}")
            print(f"Aggregated Data - Temp: {data['temperature']}, Rain: {data['rainfall']}, Humidity: {data['humidity']}")

            if (temp_min <= data['temperature'] <= temp_max and
                hum_min <= data['humidity'] <= hum_max):
                print("-> Crop matches the conditions.\n")
                crop_with_season = crop.copy()
                crop_with_season['season'] = season
                matching_crops.append(crop_with_season)
            else:
                print("-> Crop does not match the conditions.\n")

        print(f"Total Matching Crops for Season ({season}): {len(matching_crops)}\n")
        # Select top N crops for the season (now top 3)
        top_crops = matching_crops[:3]
        suggested_crops.extend(top_crops)

    print(f"Total Suggested Crops: {len(suggested_crops)}\n")
    return suggested_crops

# =========================
# API Endpoints
# =========================

# Crop Suggestion API Endpoint
@router.post("/suggest_crops", response_model=List[CropSuggestion])
async def get_crop_suggestions(request: ForecastRequest):
    try:
        # Parse dates
        print(request.start_date)
        print(request.end_date)
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)

        # Validate date range
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date.")

        # Forecast weather data
        forecast_df = forecast_weather(start_date, end_date)
        
        # Aggregate data by season
        season_aggregates = aggregate_forecast_by_season(forecast_df)

        # Suggest crops
        suggested_crops = suggest_crops(season_aggregates)

        if not suggested_crops:
            return []

        # Prepare response
        response = []
        for crop in suggested_crops:
            response.append(CropSuggestion(
                name=crop['name'],
                # image_link=crop.get('image_link', ''),  # Uncomment if you have image links
                season=crop['season']
            ))
        return response

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

# Disease Analysis Endpoint
@router.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    try:
        # Read image file
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert('RGB')

        # Get disease prediction
        disease_name = model.predict(img)

        # Draw red square around infected area
        img_with_square = draw_red_square(img)

        # Convert image to base64
        buffered = BytesIO()
        img_with_square.save(buffered, format="JPEG")
        encoded_image = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        # Get cure recommendations
        cure_recommendations = get_cure_recommendations(disease_name)

        return JSONResponse(content={
            "processedImage": encoded_image,
            "diseaseName": disease_name,
            "cure": cure_recommendations
        })
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Chatbot Endpoint
# @router.post("/chatbot")
# async def chatbot(request: Request):
#     try:
#         data = await request.json()
#         question = data.get("question")

#         if not question:
#             return JSONResponse(content={"error": "No question provided."}, status_code=400)

#         # Add the question to the prompt
#         prompt = prompt_template.format(question=question)

#         # Generate a response using the text-generation model
#         response = chat_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']

#         # Extract the generated answer after the prompt (removing the prompt text)
#         generated_answer = response.split("A:")[-1].strip()

#         return JSONResponse(content={"answer": generated_answer})
#     except Exception as e:
#         print(f"Error: {e}")
#         traceback.print_exc()
#         return JSONResponse(content={"error": str(e)}, status_code=500)
