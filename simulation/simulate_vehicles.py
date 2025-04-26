import pandas as pd
import numpy as np
from lxml import etree
import traci
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import json

# CONFIGURATION

# List of Junction IDs to process
JUNCTION_IDS_TO_PROCESS = ["4", "9", "8", "5"]
SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"
FLOW_DURATION = 60
TRIPINFO_FILE = "simulation/output/tripinfo.xml"

# Final enhanced CSV
ROUTE_CSV = "data/final_with_all_data.csv"


def get_car_distributions(routes, N=10, update_diffs=None, previous_probs=None):
    if previous_probs is not None:
        p = previous_probs
    else:
        w = np.array([routes[route]["target_duration"] - routes[route]["duration_without_traffic"] for route in routes])
        w = np.clip(w, 0, None)
        p = w / w.sum()
        p = np.power(p, 2)
        p = p / p.sum()

    if update_diffs:
        p = .8 * p + 0.2 * p * np.array(update_diffs)
        p = p / p.sum()

    counts = np.random.multinomial(N, p)
    route_ids = list(routes.keys())
    route_counts = list(zip(route_ids, list(map(int, map(lambda x: max(1,x), counts)))))
    return route_counts, p


# === Generate randomized flow file
def generate_flow_route_file(routes, route_cache, N, begin_time=0, update_diffs=None, previous_probs=None):
    """
    Generate a flow route file based on the routes and their respective cache.
    """
    route_counts, new_probs = get_car_distributions(routes, N, update_diffs,previous_probs)
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    flow_entries = []

    for rid, info in routes.items():
        key = (info["from_edge"], info["to_edge"])
        if key not in route_cache:
            try:
                route_cache[key] = traci.simulation.findRoute(info["from_edge"], info["to_edge"]).edges
            except Exception as e:
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

    # Sort flows by begin time
    # flow_entries.sort(key=lambda x: x["begin"])

    for entry in flow_entries:
        etree.SubElement(root, "route", id=entry["rid"], edges=" ".join(entry["edges"]))
        etree.SubElement(root, "flow",
                         id=entry["rid"],
                         type="car",
                         route=entry["rid"],
                         begin=str(entry["begin"]),
                         end=str(entry["end"]),
                         number=str(entry["count"]),
                         departPos="random",
                         arrivalPos="random")

    tree = etree.ElementTree(root)
    tree.write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    return new_probs


def calculate_means_from_file():
    if not os.path.exists(TRIPINFO_FILE):
        print(f"⚠️ Warning: {TRIPINFO_FILE} not found. Skipping mean duration calculation.")
        return {}

    # Load and parse XML
    tree = ET.parse(TRIPINFO_FILE)
    root = tree.getroot()

    # Dictionary to store durations per base route
    route_durations = defaultdict(list)

    # Iterate over all <tripinfo> elements
    for tripinfo in root.findall('tripinfo'):
        trip_id = tripinfo.attrib['id']  # e.g., route_0.0
        duration = float(tripinfo.attrib['duration'])

        # Extract base route, e.g., route_0 from route_0.0
        base_route = trip_id.split('.')[0]

        route_durations[base_route].append(duration)

    # Compute mean duration per base route
    mean_durations = {
        route: sum(durations) / len(durations)
        for route, durations in route_durations.items()
    }

    return mean_durations


def recalculate_total_count(routes):
    """
    Recalculate the total count based on the route data.
    This function should return the total count of vehicles.
    """

    mean_durations = calculate_means_from_file()

    print(f"{'Route':<10} | {'Target':<10} | {'Simulated':<12} | {'Diff':<10} | {'% Diff':<10}")
    print("-" * 60)

    total_expected = 0
    total_simulated = 0
    total_diff = 0

    for route_id, route_data in routes.items():
        expected = route_data["target_duration"]
        simulated = mean_durations.get(route_id)

        if simulated is None:
            print(f"{route_id:<10} | {'MISSING':<10}")
            continue

        diff = simulated - expected
        percent_diff = (diff / expected) * 100 if expected != 0 else float('inf')

        total_expected += expected
        total_simulated += simulated
        total_diff += abs(diff)

        print(f"{route_id:<10} | {expected:<10.2f} | {simulated:<12.2f} | {diff:<10.2f} | {percent_diff:<10.2f}")

    print("-" * 60)
    # total_diff = total_expected - total_simulated
    total_percent_diff = (total_diff / total_expected) * 100 if total_expected != 0 else float('inf')
    print(
        f"{'TOTAL':<10} | {total_expected:<10.2f} | {total_simulated:<12.2f} | {total_diff:<10.2f} | {total_percent_diff:<10.2f}")


def run_simulation_once(N, begin_time,update_diffs = None, previous_probs=None):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(1)])
    new_probs = generate_flow_route_file(routes, route_cache, N, begin_time=begin_time, update_diffs = update_diffs, previous_probs=previous_probs)
    traci.close()

    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "-r", ROUTE_FILE,
        "--tripinfo-output", TRIPINFO_FILE,
        "--start",
        "--step-length", str(1)
    ])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

    return calculate_means_from_file(), new_probs


def average_means(list_of_mean_dicts):
    """
    Input: list of mean_durations dicts per simulation run
    Output: one dict of average mean per route
    """
    all_means = defaultdict(list)

    for mean_dict in list_of_mean_dicts:
        for route, val in mean_dict.items():
            all_means[route].append(val)

    return {route: sum(vals)/len(vals) for route, vals in all_means.items()}


def compare_to_targets(mean_durations, routes):
    print(f"{'Route':<10} | {'Target':<10} |{'No traffic':<10} | {'Simulated':<12} | {'Diff':<10} | {'% Diff':<10}")
    print("-" * 60)

    total_expected = 0
    total_no_traffic = 0
    total_simulated = 0
    total_diff = 0
    update_diffs = []

    for route_id, route_data in routes.items():
        expected = route_data["target_duration"]
        simulated = mean_durations.get(route_id)
        no_traffic = route_data["duration_without_traffic"]

        if simulated is None:
            print(f"{route_id:<10} | {'MISSING':<10}")
            continue

        diff = simulated - expected
        percent_diff = (diff / expected) * 100 if expected != 0 else float('inf')
        update_diffs.append(expected/simulated)

        total_expected += expected
        total_no_traffic += no_traffic
        total_simulated += simulated
        total_diff += abs(diff)

        print(f"{route_id:<10} | {expected:<10.2f} | {no_traffic:<10.2f} | {simulated:<12.2f} | {diff:<10.2f} | {percent_diff:<10.2f}")

    print("-" * 60)
    total_percent_diff = (total_diff / total_expected) * 100 if total_expected != 0 else float('inf')
    print(f"{'TOTAL':<10} | {total_expected:<10.2f} |{total_no_traffic:<10.2f} | {total_simulated:<12.2f} | {total_diff:<10.2f} | {total_percent_diff:<10.2f}")
    return total_expected, total_simulated, total_diff, total_percent_diff, update_diffs


def run_multiple_simulations(total_count, iterations=60, step=3, update_diffs=None, previous_probs=None):
    all_runs = []

    for i in range(0, iterations, step):
        print(f"\n--- Simulation #{i // step + 1} (time step {i}) ---")
        means, new_probs = run_simulation_once(total_count,i, update_diffs, previous_probs)
        all_runs.append(means)

    averaged = average_means(all_runs)
    return averaged, new_probs


if __name__ == "__main__":

    # === Load full dataset
    df_full = pd.read_csv(ROUTE_CSV)

    junction_results = {}

    df = df_full[df_full["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
    # df = df_full
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
            "target_duration": row["duration_20250327_1830"],
            "vehicle_count": None,
            "last_duration": None,
            "converged": False,
            "duration_without_traffic": row["duration_without_traffic"]
        }

    total_count = int(0.5*int(
        sum(route["target_duration"] for route in routes.values()) -
        sum(route["duration_without_traffic"] for route in routes.values())
    ))
    total_count = max(50, total_count)

    attempt = 0
    max_attempts = 100
    update_diffs = None
    previous_probs = None

    while attempt < max_attempts:
        print(f"\n=== Attempt {attempt + 1} for Junction {JUNCTION_IDS_TO_PROCESS} with TOTAL_COUNT={total_count} ===")
        averaged, new_probs = run_multiple_simulations(total_count, iterations=63, step=3, update_diffs=update_diffs, previous_probs=previous_probs)
        previous_probs = new_probs
        total_expected, total_simulated, total_diff, percent_diff, update_diffs = compare_to_targets(averaged,
                                                                                                     routes)
        if attempt % 10 == 0:
            ratio = total_expected / total_simulated if total_simulated else 1.0
            total_count = max(1, int(total_count * ratio))
            update_diffs = None
        attempt += 1
        # Prepare the new data
        route_ids = list(routes.keys())
        probabilities = {rid: round(previous_probs[i], 4) for i, rid in enumerate(route_ids)}
        junction_results = {
            "total_count": total_count,
            "probabilities": probabilities,
            "percent_diff": percent_diff
        }

        output_json_path = "simulation/output/junction_simulation_results.json"

        should_write = True

        # Check if file exists
        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                try:
                    existing_data = json.load(f)
                    existing_percent_diff = existing_data.get("percent_diff")
                    if existing_percent_diff is not None and existing_percent_diff <= percent_diff:
                        should_write = False
                except json.JSONDecodeError:
                    pass

        # Write new data if needed
        if should_write:
            with open(output_json_path, "w") as f:
                json.dump(junction_results, f, indent=4)

        if percent_diff <= 20:
            print("Acceptable total count")
            if sum(0.7 <= x <= 1.3 for x in update_diffs) / len(update_diffs) >= 0.8:
                print("Acceptable percentage of each probability")
                break

