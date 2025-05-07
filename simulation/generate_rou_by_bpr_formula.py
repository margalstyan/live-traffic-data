import json
import os
import pandas as pd
import numpy as np
from lxml import etree
import traci

# CONFIG
INPUT_JSON = "simulation/output/bpr_results.json"
INPUT_CSV = "data/final_with_all_data.csv"
OUTPUT_ROUTE_FILE = "config/generated_bpr.rou.xml"
FLOW_DURATION = 60  # seconds
INSERTION_INTERVAL = 300  # seconds between timestamps (10 minutes)
SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"

JUNCTION_IDS_TO_PROCESS = [3]  # Filter for relevant junctions

# BPR function parameters
def bpr_volume(duration, free_flow_duration, alpha, beta):
    # Invert BPR function to solve for volume (flow)
    # duration = t0 * (1 + alpha * (volume / capacity) ** beta)
    # => volume = capacity * ((duration / t0 - 1) / alpha) ** (1 / beta)
    capacity = 1  # Assuming unit capacity, since we're calculating relative volume
    ratio = (duration / free_flow_duration - 1) / alpha
    ratio = np.maximum(ratio, 0)  # avoid negative values
    volume = capacity * (ratio ** (1 / beta))
    return volume

def find_route_edges(from_edge, to_edge):
    try:
        route = traci.simulation.findRoute(from_edge, to_edge)
        return " ".join(route.edges)
    except Exception as e:
        print(f"⚠️ Warning: Could not find route from {from_edge} to {to_edge}: {e}")
        return f"{from_edge} {to_edge}"

def process_timestamp_bpr(root, timestamp_key, timestamp_data, routes, start_time):
    parameters = timestamp_data["parameters"]
    for route_id, param in parameters.items():
        if route_id not in routes:
            print(f"⚠️ Warning: {route_id} not found in CSV routes. Skipping.")
            continue

        route_info = routes[route_id]
        t0 = route_info["duration_without_traffic"]
        t = route_info["target_durations"].get(timestamp_key)
        if pd.isna(t) or t0 == 0:
            print(f"⚠️ Skipping route {route_id} due to missing or invalid durations")
            continue

        alpha = param["alpha"]
        beta = param["beta"]
        count = bpr_volume(t, t0, alpha, beta)
        count = int(np.clip(round(count * 100), 1, 9999))  # Scale and clip

        from_edge = route_info["from_edge"]
        to_edge = route_info["to_edge"]
        edges = find_route_edges(from_edge, to_edge)

        # Create route and flow
        etree.SubElement(root, "route", id=f"{route_id}_{timestamp_key}", edges=edges)
        etree.SubElement(root, "flow",
                         id=f"flow_{route_id}_{timestamp_key}",
                         type="car",
                         route=f"{route_id}_{timestamp_key}",
                         begin=str(start_time),
                         end=str(start_time + FLOW_DURATION),
                         number=str(count),
                         departPos="random",
                         arrivalPos="random")

def generate_routes_from_bpr_json(input_json_path, input_csv_path, output_xml_path):
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

    # Load JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Load CSV and prepare routes dict
    df_full = pd.read_csv(input_csv_path)
    df = df_full[df_full["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]

    routes = {}
    for idx, row in df.iterrows():
        route_id = f"route_{idx}"
        routes[route_id] = {
            "origin": row["Origin"],
            "destination": row["Destination"],
            "from_edge": row["from_edge"],
            "to_edge": row["to_edge"],
            "duration_without_traffic": row["duration_without_traffic"],
            "target_durations": {col: row[col] for col in df.columns if col.startswith("duration_")}
        }

    # Build XML
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    # Start SUMO
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", "1"])

    # Process all timestamps
    current_start_time = 0
    for key in sorted(data.keys()):
        if key.startswith("duration_"):
            process_timestamp_bpr(root, key, data[key], routes, start_time=current_start_time)
            current_start_time += INSERTION_INTERVAL

    traci.close()

    # Save XML
    tree = etree.ElementTree(root)
    tree.write(output_xml_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"✅ Route file successfully created: {output_xml_path}")

if __name__ == "__main__":
    generate_routes_from_bpr_json(INPUT_JSON, INPUT_CSV, OUTPUT_ROUTE_FILE)
