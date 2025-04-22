import pandas as pd

# Load the main CSV
main_df = pd.read_csv("../data/result_10_minutes.csv", sep=";")

# Step 1: Add 'duration_without_traffic' column
duration_cols = [col for col in main_df.columns if col.startswith("duration_")]
main_df["duration_without_traffic"] = main_df[duration_cols].min(axis=1)

# Step 2: Add 'from_edge' and 'to_edge' columns using points_with_edge_ids.csv
edges_df = pd.read_csv("../data/points_with_edge_id.csv", sep=";")
edge_map = dict(zip(edges_df["key"], edges_df["edge_id"]))

main_df["from_edge"] = main_df["Origin"].map(edge_map)
main_df["to_edge"] = main_df["Destination"].map(edge_map)

# Step 3: Add 'distance' column using routes_with_distances.csv
distances_df = pd.read_csv("../data/routes_with_distances.csv")
dist_map = distances_df.set_index(["Origin", "Destination", "Direction"])["Distance_m"].to_dict()

main_df["distance"] = main_df.apply(
    lambda row: dist_map.get((row["Origin"], row["Destination"], row["Direction"])), axis=1
)

# Save the final enhanced CSV
main_df.to_csv("../data/final_with_all_data.csv", index=False)
