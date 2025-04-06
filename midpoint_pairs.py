import pandas as pd

# Load your files
points_df = pd.read_csv("points.csv")
traffic_df = pd.read_csv("result.csv")

# Preprocess coordinates
points_df[['latitude', 'longitude']] = points_df['coordinate'].str.split(',', expand=True).astype(float)

# Get unique connections (origin-destination pairs)
connections = traffic_df[['Origin', 'Destination']].drop_duplicates()

# Add coordinates for Origin
connections = connections.merge(
    points_df[['key', 'latitude', 'longitude']],
    left_on='Origin', right_on='key'
).rename(columns={'latitude': 'lat_a', 'longitude': 'lon_a'}).drop(columns=['key'])

# Add coordinates for Destination
connections = connections.merge(
    points_df[['key', 'latitude', 'longitude']],
    left_on='Destination', right_on='key'
).rename(columns={'latitude': 'lat_b', 'longitude': 'lon_b'}).drop(columns=['key'])

# Create readable junction ID
connections['junction_id'] = connections['Origin'] + '_to_' + connections['Destination']

# Reorder and export
final_df = connections[['junction_id', 'Origin', 'lat_a', 'lon_a', 'Destination', 'lat_b', 'lon_b']]
final_df.to_csv("midpoint_pairs.csv", index=False)
