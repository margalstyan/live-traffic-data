import pandas as pd
import requests
import time

# Load midpoint pairs
df = pd.read_csv("data/midpoint_pairs.csv")

# Your Google Maps API Key
API_KEY = "AIzaSyBJ1jHA-FbNDE4wheJxwQ3FnUK6iR9PW7o"

# Output list
results = []

# Loop through each road pair
for i, row in df.iterrows():
    origin = f"{row['lat_a']},{row['lon_a']}"
    destination = f"{row['lat_b']},{row['lon_b']}"
    junction_id = row['junction_id']

    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={origin}&destination={destination}&key={API_KEY}"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if data["status"] == "OK" and data.get("routes"):
            steps = data["routes"][0]["legs"][0]["steps"]

            # Try to find a step with a junction-like maneuver
            junction_location = None
            for step in steps:
                if "maneuver" in step and step["maneuver"] in ["turn-right", "turn-left", "merge", "roundabout-right", "roundabout-left"]:
                    junction_location = step["start_location"]
                    break

            # Fallback: use the middle step if no turn/merge was found
            if not junction_location and steps:
                mid_step = steps[len(steps) // 2]
                junction_location = mid_step["start_location"]

            if junction_location:
                results.append({
                    "junction_id": junction_id,
                    "lat_junction": junction_location["lat"],
                    "lon_junction": junction_location["lng"]
                })
                print(f"✅ Found for {junction_id}")
            else:
                print(f"⚠️  No suitable step for {junction_id}")
        else:
            print(f"❌ Failed for {junction_id} | Status: {data['status']}")
            if "error_message" in data:
                print("Reason:", data["error_message"])

    except Exception as e:
        print(f"❗ Error for {junction_id}:", str(e))

    time.sleep(0.25)  # respect rate limits

# Save to CSV
pd.DataFrame(results).to_csv("data/junctions_from_google.csv", index=False)
print("✅ All done. Saved to 'junctions_from_google.csv'")
