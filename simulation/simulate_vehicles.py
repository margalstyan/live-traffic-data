import pandas as pd
import numpy as np
from lxml import etree
import traci
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import json

# CONFIGURATION
JUNCTION_IDS_TO_PROCESS = ["3"]
SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"
FLOW_DURATION = 60
TRIPINFO_FILE = "simulation/output/tripinfo.xml"
ROUTE_CSV = "data/final_with_all_data.csv"
OUTPUT_JSON_PATH = "simulation/output/junction_simulation_results.json"

def get_car_distributions(routes, N=10, update_diffs=None, previous_probs=None):
    if previous_probs is not None and any(diff > 0.45 for diff in previous_probs):
        previous_probs = None

    if previous_probs is not None:
        p = previous_probs
    else:
        w = np.array([routes[route]["target_duration"] - routes[route]["duration_without_traffic"] for route in routes])
        w = np.clip(w, 0, None)
        p = w / w.sum()
        p = np.power(p, 2)
        p = p / p.sum()

    if update_diffs is not None:
        p = p * np.array(update_diffs)
        p = p / p.sum()

    counts = np.random.multinomial(N, p)
    route_ids = list(routes.keys())
    route_counts = list(zip(route_ids, list(map(int, map(lambda x: max(1, x), counts)))))
    return route_counts, p

def generate_flow_route_file(routes, route_cache, N, begin_time=0, update_diffs=None, previous_probs=None):
    route_counts, new_probs = get_car_distributions(routes, N, update_diffs, previous_probs)
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    flow_entries = []
    for rid, info in routes.items():
        key = (info["from_edge"], info["to_edge"])
        if key not in route_cache:
            try:
                route_cache[key] = traci.simulation.findRoute(info["from_edge"], info["to_edge"]).edges
            except Exception as e:
                print(f"Error finding route from {info['from_edge']} to {info['to_edge']}: {e}")
                continue

        edges = route_cache[key]
        begin = begin_time
        end = begin + FLOW_DURATION
        count = next((count for r, count in route_counts if r == rid), 0)

        flow_entries.append({
            "rid": rid,
            "edges": edges,
            "begin": begin,
            "end": end,
            "count": count
        })

    for entry in flow_entries:
        etree.SubElement(root, "route", id=entry["rid"], edges=" ".join(entry["edges"]))
        etree.SubElement(root, "flow", id=entry["rid"], type="car", route=entry["rid"], begin=str(entry["begin"]),
                         end=str(entry["end"]), number=str(entry["count"]), departPos="random", arrivalPos="random")

    tree = etree.ElementTree(root)
    tree.write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    return new_probs

def calculate_means_from_file():
    if not os.path.exists(TRIPINFO_FILE):
        print(f"⚠️ Warning: {TRIPINFO_FILE} not found. Skipping mean duration calculation.")
        return {}

    tree = ET.parse(TRIPINFO_FILE)
    root = tree.getroot()

    route_durations = defaultdict(list)

    for tripinfo in root.findall('tripinfo'):
        trip_id = tripinfo.attrib['id']
        duration = float(tripinfo.attrib['duration'])
        base_route = trip_id.split('.')[0]
        route_durations[base_route].append(duration)

    mean_durations = {route: sum(durations) / len(durations) for route, durations in route_durations.items()}
    return mean_durations

def run_simulation_once(routes, route_cache, N, begin_time, update_diffs=None, previous_probs=None):
    print(f"Running flow generation simulation with {N} vehicles from {begin_time} to {begin_time + FLOW_DURATION}...")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(1)])
    new_probs = generate_flow_route_file(routes, route_cache, N, begin_time=begin_time, update_diffs=update_diffs, previous_probs=previous_probs)
    traci.close()

    print("Running traffic simulation...")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE, "--tripinfo-output", TRIPINFO_FILE, "--start", "--step-length", str(1)])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

    print("Simulation completed. Calculating mean durations...")
    return calculate_means_from_file(), new_probs

def average_means(list_of_mean_dicts):
    all_means = defaultdict(list)
    for mean_dict in list_of_mean_dicts:
        for route, val in mean_dict.items():
            all_means[route].append(val)
    return {route: sum(vals) / len(vals) for route, vals in all_means.items()}

def compare_to_targets(mean_durations, routes):
    print(f"{'Route':<10} | {'Target':<10} | {'Simulated':<12} | {'Diff':<10} | {'% Diff':<10}")
    print("-" * 60)

    total_expected, total_simulated, total_diff = 0, 0, 0
    update_diffs = []

    for route_id, route_data in routes.items():
        expected = route_data["target_duration"]
        simulated = mean_durations.get(route_id)

        if simulated is None:
            print(f"{route_id:<10} | {'MISSING':<10}")
            continue

        diff = simulated - expected
        percent_diff = (diff / expected) * 100 if expected != 0 else float('inf')
        update_diffs.append(expected / simulated)

        total_expected += expected
        total_simulated += simulated
        total_diff += abs(diff)

        print(f"{route_id:<10} | {expected:<10.2f} | {simulated:<12.2f} | {diff:<10.2f} | {percent_diff:<10.2f}")

    print("-" * 60)
    total_percent_diff = (total_diff / total_expected) * 100 if total_expected != 0 else float('inf')
    print(f"{'TOTAL':<10} | {total_expected:<10.2f} | {total_simulated:<12.2f} | {total_diff:<10.2f} | {total_percent_diff:<10.2f}")

    return total_expected, total_simulated, total_diff, total_percent_diff, update_diffs

def run_multiple_simulations(routes, route_cache, total_count, iterations=60, step=3, update_diffs=None, previous_probs=None):
    all_runs = []
    for i in range(0, iterations, step):
        print(f"--- Running simulation at time step {i} ---")
        means, new_probs = run_simulation_once(routes, route_cache, total_count, i, update_diffs, previous_probs)
        all_runs.append(means)
    averaged = average_means(all_runs)
    return averaged, new_probs

if __name__ == "__main__":
    df_full = pd.read_csv(ROUTE_CSV)
    timestamp_columns = [
        col for col in df_full.columns
        if col.startswith("duration_") and col > "duration_20250327_1900"
    ]
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, "r") as f:
            try:
                junction_results = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Could not parse existing JSON. Starting fresh.")
                junction_results = {}
    else:
        junction_results = {}

    last_used_probabilities = None

    for timestamp in timestamp_columns:
        print(f"\n\n=======================\nStarting simulation for {timestamp}\n=======================\n\n")

        df = df_full[df_full["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
        ROUTE_FILE = "config/generated_flows_simultaneously.rou.xml"

        routes = {}
        route_cache = {}

        for idx, row in df.iterrows():
            route_id = f"route_{idx}"
            routes[route_id] = {
                "origin": row["Origin"],
                "destination": row["Destination"],
                "from_edge": row["from_edge"],
                "to_edge": row["to_edge"],
                "target_duration": row[timestamp],
                "vehicle_count": None,
                "last_duration": None,
                "converged": False,
                "duration_without_traffic": row["duration_without_traffic"]
            }

        total_count = int(0.5 * int(
            sum(route["target_duration"] for route in routes.values()) -
            sum(route["duration_without_traffic"] for route in routes.values())
        ))
        total_count = max(10, total_count)

        attempt = 0
        max_attempts = 30
        update_diffs = None
        if last_used_probabilities is not None:
            previous_probs = np.array([last_used_probabilities[rid] for rid in routes.keys()])
        else:
            previous_probs = None

        best_result = None

        while attempt < max_attempts:
            print(f"\n=== Attempt {attempt + 1} for timestamp {timestamp} with TOTAL_COUNT={total_count} ===")
            averaged, new_probs = run_multiple_simulations(routes, route_cache, total_count, iterations=63, step=3, update_diffs=update_diffs, previous_probs=previous_probs)
            total_expected, total_simulated, total_diff, percent_diff, update_diffs = compare_to_targets(averaged, routes)

            probabilities = {rid: round(new_probs[i], 4) for i, rid in enumerate(routes.keys())}
            single_result = {
                "total_count": total_count,
                "probabilities": probabilities,
                "percent_diff": percent_diff
            }

            if attempt == 0 or (single_result["percent_diff"] < junction_results.get(timestamp, {}).get("percent_diff", float('inf'))):
                junction_results[timestamp] = single_result
                with open(OUTPUT_JSON_PATH, "w") as f:
                    json.dump(junction_results, f, indent=4)

            if percent_diff <= 40 and sum(0.3 <= x <= 1.8 for x in update_diffs) / len(update_diffs) >= 0.8:
                print(f"✅ Acceptable error for {timestamp}. Moving to next timestamp...")
                break
            else:
                print("percent diff:", percent_diff)
                print("update params:", update_diffs)
            all_below_1 = all(x < 1 for x in update_diffs)
            all_above_1_1 = all(x > 1.1 for x in update_diffs)

            if attempt % 5 == 0 or all_below_1 or all_above_1_1:
                ratio = total_expected / total_simulated if total_simulated else 1.0
                total_count = max(1, int(total_count * ratio))
                update_diffs = np.ones(len(update_diffs))

            previous_probs = new_probs
            attempt += 1

        last_used_probabilities = junction_results[timestamp]["probabilities"]

    print("\n✅ All timestamps processed successfully!")
