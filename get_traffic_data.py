import pandas as pd
import googlemaps
from datetime import datetime, timedelta
import pytz
import time
import os

# === CONFIG ===
GOOGLE_API_KEY = 'AIzaSyDVlpHB0cKSVr9SWIAI1kisAdqtG1Hnl1A'
POINTS_CSV = 'points.csv'
ROUTES_CSV = 'routes.csv'
OUTPUT_CSV = 'data.csv'  # save final combined data here
SLEEP_TIME = 3600  # 1 hour in seconds
TOTAL_RUNS = 1

# === INIT ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Timezone for Yerevan
tz = pytz.timezone('Asia/Yerevan')

# Load points and prepare lookup
points_df = pd.read_csv(POINTS_CSV)
points_df[['lat', 'lon']] = points_df['coordinate'].str.split(',', expand=True).astype(float)
coord_dict = points_df.set_index('key')[['lat', 'lon']].to_dict('index')

# Function to get traffic durations
def get_durations(routes_df):
    durations = []
    for _, row in routes_df.iterrows():
        origin_key = row['Origin']
        destination_key = row['Destination']

        if origin_key not in coord_dict or destination_key not in coord_dict:
            print(f"Skipping missing points: {origin_key}, {destination_key}")
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

# === MAIN LOOP ===
for i in range(TOTAL_RUNS):
    print(f"\n🕒 Run {i+1}/{TOTAL_RUNS}")

    yerevan_time = datetime.now(tz)
    timestamp = yerevan_time.strftime('%Y%m%d_%H%M')
    duration_col_name = f'duration_in_traffic_{timestamp}'

    routes_df = pd.read_csv(ROUTES_CSV)
    durations_traffic = get_durations(routes_df)

    routes_df[duration_col_name] = durations_traffic

    if os.path.exists(OUTPUT_CSV):
        # Load current data and append new column
        data_df = pd.read_csv(OUTPUT_CSV)
        data_df[duration_col_name] = routes_df[duration_col_name]
    else:
        # First time: create file
        data_df = routes_df.copy()

    # Save updated file
    data_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {duration_col_name} to {OUTPUT_CSV}")

    if i < TOTAL_RUNS - 1:
        print("⏳ Waiting 1 hour for next run...")
        time.sleep(SLEEP_TIME)
