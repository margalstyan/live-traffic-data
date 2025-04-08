# import pandas as pd
# import sumolib
#
# net = sumolib.net.readNet("osm.net.xml")  # Use your actual path
#
# points_df = pd.read_csv("data/points.csv")
#
# def get_closest_edge(lat, lon):
#     x, y = net.convertLonLat2XY(lon, lat)
#     edges = net.getNeighboringEdges(x, y, 200)
#     if not edges:
#         return None
#
#     # Detect format: tuple or edge
#     if isinstance(edges[0], tuple):
#         _, closest_edge = min(edges, key=lambda e: e[0])  # e[0] is distance
#     else:
#         # Fallback if only edges are returned
#         closest_edge = edges[0]  # assume already sorted or no distance info
#
#     return closest_edge
#
# mapped = []
# for _, row in points_df.iterrows():
#     key = row["key"]
#     lat, lon = map(float, row["coordinate"].split(","))
#     edge = get_closest_edge(lat, lon)
#     mapped.append({
#         "key": key,
#         "coordinate": row["coordinate"],
#         "edgeID": edge.getID() if edge else "NOT_FOUND",
#         "direction": "reverse" if key.endswith("-") else "forward"
#     })
#
# pd.DataFrame(mapped).to_csv("points_with_edges.csv", index=False)
# print("✅ points_with_edges.csv saved!")
import pandas as pd

# Load the CSV
df = pd.read_csv('data/routes_with_edges.csv')

# Convert distance and duration columns to integers
df['distance'] = df['distance'].astype(int)
df['duration'] = df['duration'].astype(int)

# Save back to the same file (overwrite)
df.to_csv('data/routes_with_edges.csv', index=False)

print("distance and duration columns converted to integers.")
