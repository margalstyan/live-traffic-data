import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree

# Load datasets
points_df = pd.read_csv('data/points.csv')
traffic_data = pd.read_csv('data/result.csv')

# Prepare connections
connections = traffic_data[['Origin', 'Destination', 'Direction']].drop_duplicates()

# Extract coordinates
points_df[['latitude', 'longitude']] = points_df['coordinate'].str.split(',', expand=True).astype(float)

# Matching junctions
junctions_with_coords = set(points_df['key'])
connections_filtered = connections[
    connections['Origin'].isin(junctions_with_coords) &
    connections['Destination'].isin(junctions_with_coords)
]

# nodes.xml
nodes_root_final = Element('nodes')
for _, row in points_df.iterrows():
    SubElement(nodes_root_final, 'node', id=row['key'], x=str(row['longitude']), y=str(row['latitude']), type="traffic_light")
ElementTree(nodes_root_final).write('xml/nodes_final.xml', encoding='utf-8', xml_declaration=True)

# edges.xml
edges_root_final = Element('edges')
for _, row in connections_filtered.iterrows():
    edge_id = f"{row['Origin']}_to_{row['Destination']}"
    SubElement(edges_root_final, 'edge', id=edge_id, attrib={'from': row['Origin'], 'to': row['Destination']}, priority="1", numLanes="1", speed="13.89")
ElementTree(edges_root_final).write('xml/edges_final.xml', encoding='utf-8', xml_declaration=True)
