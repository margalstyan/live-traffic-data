import pandas as pd

def generate_mixed_route_data(
    routes_file="data/routes_with_edges.csv",
    duration_file="data/result_10_minutes.csv",
    output_file="mixed_route_data.csv"
):
    # Load route edge data
    routes_df = pd.read_csv(routes_file)

    # Load timestamp duration data (semicolon-separated)
    duration_df = pd.read_csv(duration_file, sep=";")

    # Merge both files on Origin, Destination, and Direction
    merged_df = pd.merge(
        routes_df,
        duration_df,
        on=["Origin", "Destination", "Direction"],
        how="inner"
    )

    # Detect all duration timestamp columns
    duration_cols = [col for col in merged_df.columns if col.startswith("duration_")]

    # Compute average speed across all timestamps
    merged_df["avg_speed"] = merged_df["distance"] / merged_df[duration_cols].mean(axis=1)

    # Optional: sort by lowest average speed first (to prioritize slow segments)
    merged_df = merged_df.sort_values(by="avg_speed")

    # Save the merged result
    merged_df.to_csv(output_file, index=False)
    print(f"[✓] File saved: {output_file}")

# Run the function
if __name__ == "__main__":
    generate_mixed_route_data()
