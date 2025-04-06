import pandas as pd
import xml.etree.ElementTree as ET
import ast
import os
from pyproj import Transformer

# === Load data ===
points_df = pd.read_csv('data/points.csv')
junctions_df = pd.read_csv('data/junctions.csv')
routes_df = pd.read_csv('data/routes.csv')
distances_df = pd.read_csv('data/road_distances.csv')

# === Prepare output directory ===
os.makedirs('xml', exist_ok=True)

# === Set up coordinate transformer (WGS84 to UTM Zone 38N for Armenia) ===
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32638", always_xy=True)

# === Parse coordinates and convert to UTM ===
points_df['coordinate'] = points_df['coordinate'].apply(lambda x: tuple(map(float, x.split(','))))
points_df[['lon', 'lat']] = pd.DataFrame(points_df['coordinate'].tolist(), index=points_df.index)
points_df[['x', 'y']] = points_df.apply(lambda row: pd.Series(transformer.transform(row['lon'], row['lat'])), axis=1)
points_df.set_index('key', inplace=True)

# === NODE creation ===
nodes = ET.Element('nodes')

# 1. Add all points from points.csv as SUMO nodes
for point_name, row in points_df.iterrows():
    ET.SubElement(nodes, 'node',
                  id=point_name,
                  x=str(row['x']),
                  y=str(row['y']),
                  type='priority')

# 2. Add junction nodes (if distinct)
for idx, row in junctions_df.iterrows():
    lat, lon = ast.literal_eval(row['coordinate'])
    x, y = transformer.transform(lon, lat)
    node_id = f"junction_{idx}"
    node_type = 'traffic_light' if row['traffic_light'] == 1 else 'priority'
    ET.SubElement(nodes, 'node',
                  id=node_id,
                  x=str(x),
                  y=str(y),
                  type=node_type)

ET.ElementTree(nodes).write('xml/nodes.xml', encoding='utf-8', xml_declaration=True)

# === EDGE creation ===
edges = ET.Element('edges')

for _, row in routes_df.iterrows():
    origin = row['Origin']
    destination = row['Destination']

    # Try to get real distance
    dist_match = distances_df[
        (distances_df['Origin'] == origin) &
        (distances_df['Destination'] == destination)
    ]
    if not dist_match.empty:
        distance = float(dist_match.iloc[0]['Distance_m'])
    else:
        coord1 = points_df.loc[origin]
        coord2 = points_df.loc[destination]
        dx = coord2['x'] - coord1['x']
        dy = coord2['y'] - coord1['y']
        distance = ((dx**2 + dy**2) ** 0.5)

    ET.SubElement(edges, 'edge', attrib={
        'id': f"{origin}_to_{destination}",
        'from': origin,
        'to': destination,
        'length': str(round(distance, 2)),
        'numLanes': '1',
        'speed': '13.9'
    })

ET.ElementTree(edges).write('xml/edges.xml', encoding='utf-8', xml_declaration=True)

# === CONNECTION creation ===
connections = ET.Element('connections')

for idx, row in junctions_df.iterrows():
    connected = [r.strip() for r in row['connected_roads'].split(',') if r.strip()]
    for i in range(len(connected)):
        for j in range(len(connected)):
            if i != j:
                ET.SubElement(connections, 'connection', attrib={
                    'from': connected[i],
                    'to': connected[j]
                })

ET.ElementTree(connections).write('xml/connections.xml', encoding='utf-8', xml_declaration=True)

# === ROUTES creation ===
routes = ET.Element('routes')
ET.SubElement(routes, 'vType', id="car", accel="2.6", decel="4.5",
              sigma="0.5", length="5", maxSpeed="13.9")

for idx, row in routes_df.iterrows():
    ET.SubElement(routes, 'trip', attrib={
        'id': f"trip_{idx}",
        'type': "car",
        'from': f"{row['Origin']}_to_{row['Destination']}",
        'to': f"{row['Origin']}_to_{row['Destination']}",
        'depart': str(idx)
    })

ET.ElementTree(routes).write('xml/routes.rou.xml', encoding='utf-8', xml_declaration=True)

print("✅ SUMO XML files generated in 'xml/' folder:")
print("- nodes.xml")
print("- edges.xml")
print("- connections.xml")
print("- routes.rou.xml")
