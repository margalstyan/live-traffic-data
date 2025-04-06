import requests

API_KEY = "AIzaSyBJ1jHA-FbNDE4wheJxwQ3FnUK6iR9PW7o"
origin = "40.1782,44.5146"
destination = "40.1775,44.5149"

url = (
    f"https://maps.googleapis.com/maps/api/directions/json?"
    f"origin={origin}&destination={destination}&key={API_KEY}"
)

response = requests.get(url)
data = response.json()

# Show whole response for debugging
print(data)

if data["status"] == "OK" and data.get("routes"):
    steps = data["routes"][0]["legs"][0]["steps"]
    for step in steps:
        print("From:", step["start_location"], "→", "To:", step["end_location"])
else:
    print("❌ No route found. Status:", data["status"])
    if "error_message" in data:
        print("🧠 Reason:", data["error_message"])
