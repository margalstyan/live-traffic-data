import pandas as pd
import xml.etree.ElementTree as ET

def parse_edges_with_junctions(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    edge_info = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        if edge_id and from_node and to_node:
            edge_info[edge_id] = {"from": from_node, "to": to_node}
    return edge_info

def map_direction(direction):
    return {
        "straight": "s",
        "left": "l",
        "right": "r"
    }.get(direction.lower(), "s")

def normalize_street_name(name):
    return name.strip().rstrip("-")

def find_edges_for_origin(edge_info, street_name):
    edges = []
    for eid, info in edge_info.items():
        if street_name in eid:
            if info["from"].startswith("mid_") and info["to"].startswith("junc"):
                edges.append((eid, info["from"], info["to"]))

    return edges
def find_edges_for_destination(edge_info, street_name):
    edges = []
    for eid, info in edge_info.items():
        if street_name in eid:
            if info["from"].startswith("junc") and info["to"].startswith("mid_"):
                edges.append((eid, info["from"], info["to"]))
    return edges

def generate_connections_from_routes_csv(edges_path, routes_path):
    edge_info = parse_edges_with_junctions(edges_path)
    routes_df = pd.read_csv(routes_path)

    connections = []

    for _, row in routes_df.iterrows():
        origin_raw = row["Origin"]
        destination_raw = row["Destination"]
        direction = row["Direction"]

        origin_name = normalize_street_name(origin_raw)
        destination_name = normalize_street_name(destination_raw)

        from_edges = find_edges_for_origin(edge_info, origin_name)
        to_edges = find_edges_for_destination(edge_info, destination_name)

        for from_edge, from_from, from_to in from_edges:
            for to_edge, to_from, to_to in to_edges:
                if from_to == to_from and from_edge != to_edge:
                    connection = {
                        "from": from_edge,
                        "to": to_edge,
                        "fromLane": "0",
                        "toLane": "0",
                        "dir": map_direction(direction),
                        "state": "M"
                    }
                    connections.append(connection)
                else:
                    # This line ensures we skip reverse direction edges
                    continue

    return connections


def write_connections(connections, output_file):
    root = ET.Element("connections")
    for conn in connections:
        ET.SubElement(root, "connection", conn)
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Usage
if __name__ == "__main__":
    edges_path = "edges.xml"
    routes_path = "../data/routes.csv"
    output_file = "connections.xml"

    connections = generate_connections_from_routes_csv(edges_path, routes_path)
    write_connections(connections, output_file)
    print(f"✅ Generated {len(connections)} valid connections to {output_file}.")
