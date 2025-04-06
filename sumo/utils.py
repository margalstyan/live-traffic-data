import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32638", always_xy=True)

def write_nodes(nodes, filename="nodes.xml"):
    root = ET.Element("nodes")
    for node in nodes:
        attrib = {
            "id": node["id"],
            "x": str(node["x"]),
            "y": str(node["y"]),
            "type": node.get("type", "priority")
        }
        ET.SubElement(root, "node", attrib)
    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)


def write_edges(edges, filename="edges.xml"):
    root = ET.Element("edges")
    for edge in edges:
        attrib = {
            "id": edge["id"],
            "from": edge["from"],
            "to": edge["to"],
            "priority": str(edge.get("priority", 1)),
            "numLanes": str(edge.get("numLanes", 1)),
            "speed": str(edge.get("speed", 13.9))  # default: 50 km/h
        }
        ET.SubElement(root, "edge", attrib)
    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)


def write_traffic_light(tl_id, tl_type="static", programID="0", offset="0", phases=None, filename="tlLogic.xml"):
    root = ET.Element("additional")

    tl = ET.SubElement(root, "tlLogic", id=tl_id, type=tl_type, programID=programID, offset=offset)

    if not phases:
        raise ValueError("You must provide at least one traffic light phase.")

    for phase in phases:
        ET.SubElement(tl, "phase", duration=str(phase["duration"]), state=phase["state"])

    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)


def write_routes(vehicle_routes, vehicle_type=None, filename="routes.xml"):
    """
    vehicle_routes: list of dicts, e.g.
        [
            {"id": "veh0", "depart": 0, "route": ["BA", "AC"]},
            {"id": "veh1", "depart": 10, "route": ["CA", "AB"]}
        ]
    vehicle_type: dict with type definition, e.g.
        {
            "id": "car", "accel": "1.0", "decel": "4.5", "sigma": "0.5",
            "length": "5", "maxSpeed": "25", "guiShape": "passenger"
        }
    """
    root = ET.Element("routes")

    # Default vehicle type
    if vehicle_type is None:
        vehicle_type = {
            "id": "car",
            "accel": "1.0",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5",
            "maxSpeed": "25",
            "guiShape": "passenger"
        }

    ET.SubElement(root, "vType", vehicle_type)

    for veh in vehicle_routes:
        vehicle = ET.SubElement(root, "vehicle", {
            "id": veh["id"],
            "type": vehicle_type["id"],
            "depart": str(veh["depart"])
        })
        ET.SubElement(vehicle, "route", {
            "edges": " ".join(veh["route"])
        })

    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)


def parse_csv_to_nodes(csv_file):
    df = pd.read_csv(csv_file)
    df[['lat', 'lon']] = df['coordinate'].str.extract(r'([\d\.]+),\s*([\d\.]+)').astype(float)

    df['x'], df['y'] = transformer.transform(df['lon'].values, df['lat'].values)
    nodes = []
    for i, row in df.iterrows():
        node = {
            "id": f"junc{i}",
            "x": str(round(row["x"], 2)),
            "y": str(round(row["y"], 2)),
            "type": "traffic_light" if row["traffic_light"] == 1 else "priority"
        }
        nodes.append(node)

    return nodes

def parse_csv_to_edges(csv_file):
    df = pd.read_csv(csv_file)
    road_base_to_nodes = defaultdict(set)

    for i, row in df.iterrows():
        node_id = f"junc{i}"
        roads = [r.strip() for r in row['connected_roads'].split(',')]
        for road in roads:
            base = road.replace("-", "")
            road_base_to_nodes[base].add((road, node_id))  # store both direction info and node

    # Step 2: create edges from matched road directions
    edges = set()

    for base, road_nodes in road_base_to_nodes.items():
        nodes_by_name = defaultdict(set)

        for road_name, node_id in road_nodes:
            nodes_by_name[road_name].add(node_id)

        # Combine all nodes, bidirectional
        all_nodes = set(n for _, n in road_nodes)

        for from_node in all_nodes:
            for to_node in all_nodes:
                if from_node != to_node:
                    edge_id = f"{from_node}_{to_node}_{base}"
                    edges.add((edge_id, from_node, to_node))
    edges = [{"id": edge[0], "from": edge[1], "to": edge[2]} for edge in edges]
    return edges

def parse_csv_to_visual_nodes(csv_file):
    midpoints_df = pd.read_csv(csv_file)  # replace with your file
    midpoints_df[['lat', 'lon']] = midpoints_df['coordinate'].str.extract(r'([\d\.]+),\s*([\d\.]+)').astype(float)
    midpoints_df['x'], midpoints_df['y'] = transformer.transform(midpoints_df['lon'], midpoints_df['lat'])

    visual_nodes = []
    for i, row in midpoints_df.iterrows():
        visual_nodes.append({
            "id": f"mid_{row['key']}",
            "x": str(round(row['x'], 2)),
            "y": str(round(row['y'], 2)),
            "type": "priority"
        })
    return visual_nodes


def parse_csv_to_midpoint_edges(csv_file):
    """
    Reads midpoint edge connections from a CSV with columns: Origin, Destination, Direction
    and returns a list of edges between corresponding midpoint nodes.
    """
    df = pd.read_csv(csv_file)
    edges = []

    for _, row in df.iterrows():
        origin_id = f"mid_{row['Origin']}"
        dest_id = f"mid_{row['Destination']}"
        edge_id = f"{origin_id}_to_{dest_id}"

        edge = {
            "id": edge_id,
            "from": origin_id,
            "to": dest_id,
            "priority": "1",
            "numLanes": "1",
            "speed": "13.9"
        }
        edges.append(edge)

    return edges


# === Add to sumo/utils.py ===

def generate_visual_connection_edges(junction_csv_path, visual_nodes):
    """
    For each node in junctions.csv, check its connected_roads.
    If any match a visual_node (mid_<road>), create a connection edge
    between the junction and the visual midpoint.
    """
    df = pd.read_csv(junction_csv_path)
    connections = []

    # Map of visual node ids for quick access
    visual_ids = {node["id"]: node for node in visual_nodes}

    for i, row in df.iterrows():
        junc_id = f"junc{i}"
        roads = [r.strip() for r in row['connected_roads'].split(',')]
        for road in roads:
            base = road.replace("-", "")
            visual_node_id = f"mid_{base}"
            if visual_node_id in visual_ids:
                connections.append({
                    "id": f"{junc_id}_to_{visual_node_id}",
                    "from": junc_id,
                    "to": visual_node_id,
                    "priority": "1",
                    "numLanes": "1",
                    "speed": "13.9"
                })
                connections.append({
                    "id": f"{visual_node_id}_to_{junc_id}",
                    "from": visual_node_id,
                    "to": junc_id,
                    "priority": "1",
                    "numLanes": "1",
                    "speed": "13.9"
                })

    return connections

# === Update sumo/utils.py ===

def generate_visual_edges_and_nodes(junction_csv_path, points_csv_path):
    """
    1. Parse midpoints from points.csv
    2. For each junction from junctions.csv, check connected_roads.
    3. If a road base matches a midpoint, connect the junc <-> midpoint.
    """
    df_junc = pd.read_csv(junction_csv_path)
    df_mid = pd.read_csv(points_csv_path)

    df_mid[['lat', 'lon']] = df_mid['coordinate'].str.extract(r'([\d\.]+),\s*([\d\.]+)').astype(float)
    df_mid['x'], df_mid['y'] = transformer.transform(df_mid['lon'], df_mid['lat'])

    visual_nodes = []
    visual_node_map = {}

    for _, row in df_mid.iterrows():
        node_id = f"mid_{row['key']}"
        visual_nodes.append({
            "id": node_id,
            "x": str(round(row['x'], 2)),
            "y": str(round(row['y'], 2)),
            "type": "priority"
        })
        visual_node_map[row['key']] = node_id

    connection_edges = []

    for i, row in df_junc.iterrows():
        junc_id = f"junc{i}"
        roads = [r.strip() for r in row['connected_roads'].split(',')]
        for road in roads:
            base = road.replace("-", "")
            vis_id = visual_node_map.get(base)
            if vis_id:
                connection_edges.append({
                    "id": f"{junc_id}_to_{vis_id}",
                    "from": junc_id,
                    "to": vis_id,
                    "priority": "1",
                    "numLanes": "1",
                    "speed": "13.9"
                })
                connection_edges.append({
                    "id": f"{vis_id}_to_{junc_id}",
                    "from": vis_id,
                    "to": junc_id,
                    "priority": "1",
                    "numLanes": "1",
                    "speed": "13.9"
                })

    return visual_nodes, connection_edges

def generate_connections_from_routes(routes_csv_path, valid_edges):
    direction_map = {"Left": "l", "Straight": "s", "Right": "r"}
    df = pd.read_csv(routes_csv_path)
    connections = []

    for _, row in df.iterrows():
        from_edge = row["Origin"]
        to_edge = row["Destination"]
        direction = direction_map.get(row["Direction"].strip(), "")

        if direction == "":
            print(f"⚠️ Unrecognized direction '{row['Direction']}'")

        if from_edge not in valid_edges or to_edge not in valid_edges:
            print(f"🚫 Skipping unknown edge(s): {from_edge} → {to_edge}")
            continue

        connections.append({
            "from": from_edge,
            "to": to_edge,
            "fromLane": "0",
            "toLane": "0",
            "dir": direction,
            "state": "G"
        })

    return connections

def write_connections_to_xml(connections, output_file="connections.xml"):
    root = ET.Element("connections")

    for conn in connections:
        if conn["dir"]:  # Only write valid direction
            ET.SubElement(root, "connection", {
                "from": conn["from"],
                "to": conn["to"],
                "fromLane": conn["fromLane"],
                "toLane": conn["toLane"],
                "dir": conn["dir"],
                "state": conn["state"]
            })

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)