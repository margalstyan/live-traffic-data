import pandas as pd
import xml.etree.ElementTree as ET

# Load your CSV data
junctions_df = pd.read_csv('data/junctions.csv')
points_df = pd.read_csv('data/points.csv')
road_distances_df = pd.read_csv('data/road_distances.csv')
results_df = pd.read_csv('data/result.csv')

# Nodes.xml creation (junctions)
nodes = ET.Element('nodes')
for idx, row in junctions_df.iterrows():
    coord = eval(row['coordinate'])  # (lat, lon)
    junction_id = f"junction_{idx}"
    ET.SubElement(nodes, 'node', id=junction_id, x=str(coord[1]), y=str(coord[0]),
                  type='traffic_light' if row['traffic_light'] == 1 else 'priority')

nodes_tree = ET.ElementTree(nodes)
nodes_tree.write('xml/nodes.xml', encoding='utf-8', xml_declaration=True)

# Edges.xml creation (roads)
edges = ET.Element('edges')
for _, row in road_distances_df.iterrows():
    ET.SubElement(edges, 'edge', id=f"{row['point_a']}_to_{row['point_b']}", fromNode=str(row['point_a']),
                  toNode=str(row['point_b']), length=str(row['distance']), numLanes='1', speed='13.9')  # approx. 50 km/h

edges_tree = ET.ElementTree(edges)
edges_tree.write('xml/edges.xml', encoding='utf-8', xml_declaration=True)

# Connections.xml creation (only between roads that intersect at junctions)
connections = ET.Element('connections')

for idx, row in junctions_df.iterrows():
    connected_roads = [r.strip() for r in row['connected_roads'].split(',') if r.strip() != '']
    for i in range(len(connected_roads)):
        for j in range(len(connected_roads)):
            if i != j:
                ET.SubElement(connections, 'connection', fromEdge=connected_roads[i], toEdge=connected_roads[j])

connections_tree = ET.ElementTree(connections)
connections_tree.write('xml/connections.xml', encoding='utf-8', xml_declaration=True)

# Routes.rou.xml creation (basic example)
routes = ET.Element('routes')

# Defining vehicle types
ET.SubElement(routes, 'vType', id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", maxSpeed="13.9")

# Creating trips based on results.csv travel times
for idx, row in results_df.iterrows():
    ET.SubElement(routes, 'trip', id=f"trip_{idx}", type="car",
                  fromEdge=f"{row['start_point']}_to_{row['end_point']}",
                  toEdge=f"{row['end_point']}_to_{row['start_point']}", depart=str(idx))

routes_tree = ET.ElementTree(routes)
routes_tree.write('xml/routes.rou.xml', encoding='utf-8', xml_declaration=True)

print("XML files (nodes.xml, edges.xml, connections.xml, routes.rou.xml) have been created.")