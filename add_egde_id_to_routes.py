import pandas as pd

# Load the files
routes_df = pd.read_csv('data/routes_with_distances.csv')
points_df = pd.read_csv('data/points_with_edge_id.csv', sep=';')

# Create a mapping from key to edge_id
key_to_edge = dict(zip(points_df['key'], points_df['edge_id']))

# Map the Origin and Destination in routes_df to edge_id values
routes_df['edge_from_id'] = routes_df['Origin'].map(key_to_edge)
routes_df['edge_to_id'] = routes_df['Destination'].map(key_to_edge)

# Save the result
routes_df.to_csv('data/routes_with_edges.csv', index=False)

print("Updated file saved as 'routes_with_edges.csv'")
