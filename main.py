from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import googlemaps
from datetime import datetime
import pytz
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
# === Config ===
GOOGLE_API_KEY = "AIzaSyDVlpHB0cKSVr9SWIAI1kisAdqtG1Hnl1A"
POINTS_CSV = "data/points.csv"
ROUTES_CSV = "data/routes.csv"
OUTPUT_CSV = "data/result.csv"

tz = pytz.timezone("Asia/Yerevan")
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
scheduler = AsyncIOScheduler()
points_df = pd.read_csv(POINTS_CSV)
points_df[['lat', 'lon']] = points_df['coordinate'].str.split(',', expand=True).astype(float)
coord_dict = points_df.set_index('key')[['lat', 'lon']].to_dict('index')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Update logic ===
def get_durations(routes_df):
    durations = []
    for _, row in routes_df.iterrows():
        origin_key = row['Origin']
        destination_key = row['Destination']

        if origin_key not in coord_dict or destination_key not in coord_dict:
            durations.append(None)
            continue

        origin = coord_dict[origin_key]
        destination = coord_dict[destination_key]

        try:
            matrix = gmaps.distance_matrix(
                origins=[(origin['lat'], origin['lon'])],
                destinations=[(destination['lat'], destination['lon'])],
                mode="driving",
                departure_time='now',
                traffic_model="best_guess"
            )
            element = matrix['rows'][0]['elements'][0]
            duration_traffic = element.get('duration_in_traffic', {}).get('value')
            durations.append(duration_traffic)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error for {origin_key} → {destination_key}: {e}")
            durations.append(None)
    return durations

def run_updater():
    print("🔁 Running updater...")
    timestamp = datetime.now(tz).strftime('%Y%m%d_%H%M')
    duration_col = f'duration_{timestamp}'

    routes_df = pd.read_csv(ROUTES_CSV)
    print(f"🔁 Calculating durations for {len(routes_df)} routes...")
    durations = get_durations(routes_df)
    routes_df[duration_col] = durations

    if os.path.exists(OUTPUT_CSV):
        data_df = pd.read_csv(OUTPUT_CSV)
        data_df[duration_col] = routes_df[duration_col]
    else:
        data_df = routes_df.copy()

    data_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {duration_col} to {OUTPUT_CSV}")

@app.get("/", response_class=HTMLResponse)
async def read_data(request: Request):
    if not os.path.exists(OUTPUT_CSV):
        return templates.TemplateResponse("index.html", {"request": request, "table": "No data yet."})

    df = pd.read_csv(OUTPUT_CSV)
    return templates.TemplateResponse("index.html", {"request": request, "table": df.to_html(index=False)})


@app.post("/update")
async def stop_scheduler(background_tasks: BackgroundTasks):
    run_updater()
    return {"status": "starting update"}


@app.get("/download")
async def download_csv():
    if os.path.exists(OUTPUT_CSV):
        return FileResponse(OUTPUT_CSV, media_type='text/csv', filename="result.csv")
    return {"error": "File not found"}

@app.get("keep_alive")
async def keep_alive():
    return {"status": "running"}