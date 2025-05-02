import json
import os
import pandas as pd
import numpy as np
from lxml import etree
import traci

# CONFIG
INPUT_JSON = "../simulation/output/junction_simulation_results.json"
INPUT_CSV = "../data/final_with_all_data.csv"
OUTPUT_ROUTE_FILE = "../sumo_rl_single/routes.rou.xml"
FLOW_DURATION = 60  # seconds
INSERTION_INTERVAL = 600  # seconds between timestamps (10 minutes)
SUMO_BINARY = "sumo"
SUMO_CONFIG = "../sumo_rl_single/osm.sumocfg"

JUNCTION_IDS_TO_PROCESS = [3]  # Same as before

# Global generator for timestamp processing
timestamp_generator = None

def find_route_edges(from_edge, to_edge):
    try:
        route = traci.simulation.findRoute(from_edge, to_edge)
        return " ".join(route.edges)
    except Exception as e:
        print(f"⚠️ Warning: Could not find route from {from_edge} to {to_edge}: {e}")
        return f"{from_edge} {to_edge}"

def process_timestamp(root, timestamp_key, timestamp_data, routes, start_time):
    total_count = timestamp_data["total_count"]
    probabilities = timestamp_data["probabilities"]

    route_ids = list(probabilities.keys())
    probs_array = np.array([probabilities[rid] for rid in route_ids])
    probs_array = probs_array / probs_array.sum()

    counts = np.random.multinomial(total_count, probs_array)
    counts = np.maximum(counts, 1)

    for route_id, count in zip(route_ids, counts):
        if route_id not in routes:
            print(f"⚠️ Warning: {route_id} not found in CSV routes. Skipping.")
            continue

        route_info = routes[route_id]
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

def timestamp_generator_function(data):
    sorted_keys = sorted([key for key in data.keys() if key.startswith("duration_")])
    for key in sorted_keys:
        yield key, data[key]

def generate_routes_for_next_timestamp(input_json_path=INPUT_JSON, input_csv_path=INPUT_CSV):
    global timestamp_generator

    # Load and cache data once
    if not hasattr(generate_routes_for_next_timestamp, "data"):
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
        with open(input_json_path, "r") as f:
            generate_routes_for_next_timestamp.data = json.load(f)

    # Reset generator if exhausted
    if timestamp_generator is None:
        timestamp_generator = timestamp_generator_function(generate_routes_for_next_timestamp.data)

    try:
        timestamp_key, timestamp_data = next(timestamp_generator)
    except StopIteration:
        print("🔁 All timestamps processed. Restarting from the beginning.")
        timestamp_generator = timestamp_generator_function(generate_routes_for_next_timestamp.data)
        timestamp_key, timestamp_data = next(timestamp_generator)

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
            "target_duration": row["duration_20250327_1830"],
            "vehicle_count": None,
            "last_duration": None,
            "converged": False,
            "duration_without_traffic": row["duration_without_traffic"]
        }

    output_xml_path = OUTPUT_ROUTE_FILE

    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")
    if traci.isLoaded():
        traci.close()
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", "1"])

    process_timestamp(root, timestamp_key, timestamp_data, routes, start_time=0)

    traci.close()

    tree = etree.ElementTree(root)
    tree.write(output_xml_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print(f"✅ Route file for {timestamp_key} created: {output_xml_path}")

if __name__ == "__main__":
    generate_routes_for_next_timestamp(INPUT_JSON, INPUT_CSV)
    generate_routes_for_next_timestamp(INPUT_JSON, INPUT_CSV)
