import pandas as pd
import time
import requests

# Load CSVs
points_df = pd.read_csv("../data/points.csv")
result_df = pd.read_csv("../data/result.csv")

# Replace with your real Google API key
API_KEY = "AIzaSyBJ1jHA-FbNDE4wheJxwQ3FnUK6iR9PW7o"

# Convert coordinates to lookup dictionary
def parse_coordinates(coord_str):
    lat, lon = map(float, coord_str.strip().split(','))
    return f"{lat},{lon}"

coordinates = {
    row['key']: parse_coordinates(row['coordinate']) for _, row in points_df.iterrows()
}
reverse_coordinates = {v: k for k, v in coordinates.items()}

# Get all unique origin-destination pairs
pairs = result_df[['Origin', 'Destination']].drop_duplicates().values.tolist()

# Prepare batches
batch_size = 10
results = []

for i in range(0, len(pairs), batch_size):
    batch = pairs[i:i + batch_size]

    origins = []
    destinations = []

    for origin, dest in batch:
        if origin in coordinates and dest in coordinates:
            origins.append(coordinates[origin])
            destinations.append(coordinates[dest])

    if not origins or not destinations:
        continue

    origins_str = "|".join(origins)
    destinations_str = "|".join(destinations)

    url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={origins_str}"
        f"&destinations={destinations_str}"
        f"&key={API_KEY}"
    )

    print(f"Requesting: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print("Request failed:", response.status_code)
        continue

    data = response.json()
    for o_idx, o_coord in enumerate(origins):
        for d_idx, d_coord in enumerate(destinations):
            try:
                element = data['rows'][o_idx]['elements'][d_idx]
                distance = element['distance']['value']  # in meters
                duration = element['duration']['value']  # in seconds
                results.append({
                    'Origin': reverse_coordinates[o_coord],
                    'Destination': reverse_coordinates[d_coord],
                    'Distance_m': distance,
                    'Duration_s': duration
                })
            except Exception as e:
                print("Error parsing response:", e)

    time.sleep(1)  # respect rate limits

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("data/road_distances.csv", index=False)
print("✅ Saved to road_distances.csv")
