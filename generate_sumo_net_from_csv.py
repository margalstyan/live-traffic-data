import pandas as pd
from pyproj import Transformer
from xml.etree.ElementTree import Element, SubElement, ElementTree

# === 1. Load input files ===
points_df = pd.read_csv("data/points.csv")
traffic_df = pd.read_csv("data/result.csv")
junctions_df = pd.read_csv("data/junctions_from_google.csv")

# === 2. Convert coordinates ===

# Split lat/lon and convert to float for midpoints
points_df[['latitude', 'longitude']] = points_df['coordinate'].str.split(',', expand=True).astype(float)

# Create UTM transformer for Yerevan (Zone 38N)
transformer = Transformer.from_crs("epsg:4326", "epsg:32638", always_xy=True)

# Convert midpoints to UTM
points_df["x"], points_df["y"] = transformer.transform(points_df["longitude"].values, points_df["latitude"].values)

# Convert junctions to UTM
junctions_df["x"], junctions_df["y"] = transformer.transform(
    junctions_df["lon_junction"].values,
    junctions_df["lat_junction"].values
)

# === 3. Generate nodes.xml ===

nodes_root = Element('nodes')

# Add midpoint nodes
for _, row in points_df.iterrows():
    SubElement(
        nodes_root,
        'node',
        id=row['key'],
        x=str(row['x']),
        y=str(row['y']),
        type="priority"
    )

# Add junction nodes
for _, row in junctions_df.iterrows():
    SubElement(
        nodes_root,
        'node',
        id=row['junction_id'],
        x=str(row['x']),
        y=str(row['y']),
        type="traffic_light"
    )

# Save nodes.xml
nodes_path = "xml/combined_nodes.xml"
ElementTree(nodes_root).write(nodes_path, encoding='utf-8', xml_declaration=True)
print(f"✅ Saved: {nodes_path}")

# === 4. Generate edges.xml ===

edges_root = Element('edges')

# Extract unique Origin-Destination pairs
connections = traffic_df[['Origin', 'Destination']].drop_duplicates()

# Create junction ID column
connections['junction_id'] = connections['Origin'] + '_to_' + connections['Destination']

# Only use junctions that exist in Google CSV
valid_junctions = set(junctions_df['junction_id'])
connections = connections[connections['junction_id'].isin(valid_junctions)]

# Add edges: Origin → Junction, Destination → Junction
for _, row in connections.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    junction = row['junction_id']

    # Edge from origin midpoint to junction
    SubElement(
        edges_root,
        'edge',
        id=f"{origin}_to_{junction}",
        attrib={'from': origin, 'to': junction},
        priority="1",
        numLanes="1",
        speed="13.89"  # ~50 km/h
    )

    # Edge from destination midpoint to junction
    SubElement(
        edges_root,
        'edge',
        id=f"{destination}_to_{junction}",
        attrib={'from': destination, 'to': junction},
        priority="1",
        numLanes="1",
        speed="13.89"
    )

# Save edges.xml
edges_path = "xml/combined_edges.xml"
ElementTree(edges_root).write(edges_path, encoding='utf-8', xml_declaration=True)
print(f"✅ Saved: {edges_path}")
