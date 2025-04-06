# Let's fix the coordinate issue by converting lat-lon to UTM coordinates

# Import necessary libraries
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree
from pyproj import Transformer

# Load datasets again
points_df = pd.read_csv('data/points.csv')
traffic_data = pd.read_csv('data/result.csv')

# Prepare connections again
connections = traffic_data[['Origin', 'Destination', 'Direction']].drop_duplicates()

# Extract coordinates
points_df[['latitude', 'longitude']] = points_df['coordinate'].str.split(',', expand=True).astype(float)

# Coordinate conversion from WGS84 (lat/lon) to UTM (meters)
transformer = Transformer.from_crs("epsg:4326", "epsg:32638", always_xy=True)  # UTM zone 38N for Yerevan

# Apply transformation to each coordinate
points_df['x'], points_df['y'] = transformer.transform(points_df['longitude'].values, points_df['latitude'].values)

# Identify matching junctions (present in both datasets)
junctions_with_coords = set(points_df['key'])
connections_filtered = connections[connections['Origin'].isin(junctions_with_coords) & connections['Destination'].isin(junctions_with_coords)]

# Generate geographically accurate nodes XML with UTM coordinates
nodes_root_utm = Element('nodes')
for _, row in points_df.iterrows():
    SubElement(
        nodes_root_utm,
        'node',
        id=row['key'],
        x=str(row['x']),
        y=str(row['y']),
        type="traffic_light"
    )
nodes_tree_utm = ElementTree(nodes_root_utm)
nodes_tree_utm.write('nodes_utm.xml', encoding='utf-8', xml_declaration=True)

# Generate consistent edges XML (edges_final.xml remains same logic)
edges_root_utm = Element('edges')
for _, row in connections_filtered.iterrows():
    edge_id = f"{row['Origin']}_to_{row['Destination']}"
    SubElement(
        edges_root_utm,
        'edge',
        id=edge_id,
        attrib={'from': row['Origin'], 'to': row['Destination']},
        priority="1",
        numLanes="1",
        speed="13.89"
    )
edges_tree_utm = ElementTree(edges_root_utm)
edges_tree_utm.write('edges_utm.xml', encoding='utf-8', xml_declaration=True)
