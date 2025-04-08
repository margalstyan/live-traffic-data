import pandas as pd

# Load both CSV files
routes_df = pd.read_csv('data/routes.csv')
distances_df = pd.read_csv('data/road_distances.csv')

# Keep only the first match for each Origin + Destination pair
distances_df_unique = distances_df.drop_duplicates(subset=['Origin', 'Destination'])

# Merge with routes_df using only the first match
merged_df = routes_df.merge(
    distances_df_unique[['Origin', 'Destination', 'Distance_m', 'Duration_s']],
    on=['Origin', 'Destination'],
    how='left'
)

# Save the result
merged_df.to_csv('data/routes_with_distances.csv', index=False)

print("Merged file saved as 'routes_with_distances.csv'")
