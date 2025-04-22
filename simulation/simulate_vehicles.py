import pandas as pd
import numpy as np
from lxml import etree
import traci
import xml.etree.ElementTree as ET
from collections import defaultdict
import os

# CONFIGURATION

# List of Junction IDs to process
JUNCTION_IDS_TO_PROCESS = ["0", "1"]
SUMO_BINARY = "sumo"
SUMO_CONFIG = "../config/osm.sumocfg"
FLOW_DURATION = 60
TRIPINFO_FILE = "output/tripinfo.xml"

# Final enhanced CSV
ROUTE_CSV = "../data/final_with_all_data.csv"


def get_car_distributions(routes, N=10,update_diffs=None):
    """
    Generate car distributions based on the given route data.
    This function should return a distribution of car counts for each route.
    """
    w = np.array([routes[route]["target_duration"] - routes[route]["duration_without_traffic"] for route in routes])
    # print("weights before normalization", w)
    w = np.clip(w, 0, None)  # Ensure no negative weights
    p = w / w.sum()
    # print("weights after 1st normalization", p)

    p = np.power(p, 2)  # Square the weights to emphasize larger differences
    # print("weights after ^2", p)

    p = p / p.sum()
    print("weights before multiplying", p)

    if update_diffs:
        # print("updating percents inside function",update_diffs)
        p = p * np.array(update_diffs)
        # print("updated probabilities",p)
        p = p / p.sum()
    print("weights: ", p)

    counts = np.random.multinomial(N, p)
    route_ids = list(routes.keys())
    route_counts = list(zip(route_ids, list(map(int,counts))))
    print("generated rout counts:", route_counts)

    return route_counts


# === Generate randomized flow file
def generate_flow_route_file(routes, route_cache, N, update_diffs=None):
    """
    Generate a flow route file based on the routes and their respective cache.
    """
    route_counts = get_car_distributions(routes, N,update_diffs)
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
        begin = 0
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


def run_simulation_once(N,update_diffs = None):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(1)])
    generate_flow_route_file(routes, route_cache, N,update_diffs = update_diffs)
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

    return calculate_means_from_file()


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


def run_multiple_simulations(total_count, iterations=60, step=3, update_diffs=None):
    all_runs = []

    for i in range(0, iterations, step):
        print(f"\n--- Simulation #{i // step + 1} (time step {i}) ---")
        means = run_simulation_once(total_count, update_diffs)
        all_runs.append(means)

    averaged = average_means(all_runs)
    return averaged


if __name__ == "__main__":

    # === Load full dataset
    df_full = pd.read_csv(ROUTE_CSV)

    # === Loop through each junction
    for junction_id in JUNCTION_IDS_TO_PROCESS:
        print(f"\n=== Processing Junction ID: {junction_id} ===")
        df = df_full[df_full["Junction_id"] == junction_id]
        ROUTE_FILE = f"../config/generated_flows_{junction_id}.rou.xml"

        # Create route dict for the current junction
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

        max_attempts = 50
        attempt = 0
        update_diffs = None
        total_count = 0.3*int(
            sum(route["target_duration"] for route in routes.values()) -
            sum(route["duration_without_traffic"] for route in routes.values())
        )
        total_count = max(50, total_count)

        while attempt < max_attempts:
            print(f"\n=== Attempt {attempt + 1} with TOTAL_COUNT={total_count} ===")
            averaged = run_multiple_simulations(total_count, iterations=90, step=3, update_diffs=update_diffs)
            total_expected, total_simulated, total_diff, percent_diff, update_diffs = compare_to_targets(averaged, routes)
            print("update diffs inside while loop", update_diffs)
            print(f"\nSimulated Total: {total_simulated:.2f}, Target Total: {total_expected:.2f}, "
                  f"Diff: {total_diff:.2f}, Percent Diff: {percent_diff:.2f}%")

            if percent_diff <= 20:
                print("Total error is acceptable, checking for each route")
                if not any(x <= 0.3 or x >= 1.9 for x in update_diffs):
                    print("each route error is acceptable, breaking the loop")
                    break
                else:
                    print("there are routes with not acceptable time difference")

            # Adjust total_count proportionally
            if attempt % 5 == 0:
                ratio = total_expected / total_simulated if total_simulated != 0 else 1.0
                total_count = int(total_count * ratio )
                total_count = max(1, total_count)
                update_diffs = None
                print("Updating total count, skipping distribution update")
            else:
                print("Updating distribution, skipping update of total count")
            attempt += 1
