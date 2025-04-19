# STEPS
# calculate differences between duration and duration_without_traffic
# calculate sum of all ti_s
# calculate differences of all ti_- ti_without_traffic
# calculate a distribution of the differences
# for i in range(0, 90, 5):
#  get samples of distribution based on total sum of ti_s
#  get the mean of each route time
# compare with given duration the mean of means
import pandas as pd
import numpy as np
from lxml import etree
import traci
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

from torch_rl_sumo.utils import FLOW_DURATION

SUMO_BINARY = "sumo"
SUMO_CONFIG = "osm.sumocfg"
ROUTE_CSV = "traffic_calibration/road_load.csv"
ROUTE_FILE = "generated_flows.rou.xml"
FLOW_DURATION = 60
TRIPINFO_FILE = "tripinfo.xml"
TOTAL_COUNT = 50

# === Load CSV and prepare routes
df = pd.read_csv(ROUTE_CSV)
routes = {}
route_cache = {}

for idx, row in df.iterrows():
    route_id = f"route_{idx}"
    routes[route_id] = {
        "origin": row["Origin"],
        "destination": row["Destination"],
        "from_edge": row["from_edge"],
        "to_edge": row["to_edge"],
        "target_duration": row["duration_seconds"],
        "vehicle_count": None,
        "last_duration": None,
        "converged": False,
        "duration_without_traffic": row["duration_without_traffic"]
    }


def get_car_distributions(routes, N=10):
    """
    Generate car distributions based on the given route data.
    This function should return a distribution of car counts for each route.
    """
    w = np.array([routes[route]["target_duration"] - routes[route]["duration_without_traffic"] for route in routes])
    w = np.clip(w, 0, None)  # Ensure no negative weights
    p = w / w.sum()
    # print(f"Route probabilities 1: {p}")

    p = np.power(p, 2)  # Square the weights to emphasize larger differences

    # p = np.where(p < 0.1, p * 1.5, p)
    p = p / p.sum()
    # print(f"Route probabilities: {p}")
    counts = np.random.multinomial(N, p)
    route_ids = list(routes.keys())
    route_counts = list(zip(route_ids, list(map(int,counts))))

    return route_counts


# === Generate randomized flow file
def generate_flow_route_file(routes, route_cache, N=TOTAL_COUNT):
    """
    Generate a flow route file based on the routes and their respective cache.
    """
    route_counts = get_car_distributions(routes, N)
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

    # Print or save results
    # for route, mean_duration in sorted(mean_durations.items()):
    #     print(f"{route}: {mean_duration:.2f} seconds")

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


def run_simulation_once():
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(1)])
    generate_flow_route_file(routes, route_cache, N=TOTAL_COUNT)
    traci.close()

    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE, "--start", "--step-length", str(1)])
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

    for route_id, route_data in routes.items():
        expected = route_data["target_duration"]
        simulated = mean_durations.get(route_id)
        no_traffic = route_data["duration_without_traffic"]

        if simulated is None:
            print(f"{route_id:<10} | {'MISSING':<10}")
            continue

        diff = simulated - expected
        percent_diff = (diff / expected) * 100 if expected != 0 else float('inf')

        total_expected += expected
        total_no_traffic += no_traffic
        total_simulated += simulated
        total_diff += abs(diff)

        print(f"{route_id:<10} | {expected:<10.2f} | {no_traffic:<10.2f} | {simulated:<12.2f} | {diff:<10.2f} | {percent_diff:<10.2f}")

    print("-" * 60)
    total_percent_diff = (total_diff / total_expected) * 100 if total_expected != 0 else float('inf')
    print(f"{'TOTAL':<10} | {total_expected:<10.2f} |{total_no_traffic:<10.2f} | {total_simulated:<12.2f} | {total_diff:<10.2f} | {total_percent_diff:<10.2f}")
    return total_expected, total_simulated, total_diff, total_percent_diff


def run_multiple_simulations(total_count, iterations=60, step=3):
    global TOTAL_COUNT
    TOTAL_COUNT = total_count  # update global var used in flow generation
    all_runs = []

    for i in range(0, iterations, step):
        print(f"\n--- Simulation #{i // step + 1} (time step {i}) ---")
        means = run_simulation_once()
        all_runs.append(means)

    averaged = average_means(all_runs)
    return averaged


if __name__ == "__main__":
    max_attempts = 10
    attempt = 0
    total_count = 100  # initial value

    while attempt < max_attempts:
        print(f"\n=== Attempt {attempt + 1} with TOTAL_COUNT={total_count} ===")
        averaged = run_multiple_simulations(total_count, iterations=90, step=3)
        total_expected, total_simulated, total_diff, percent_diff = compare_to_targets(averaged, routes)

        print(f"\nSimulated Total: {total_simulated:.2f}, Target Total: {total_expected:.2f}, "
              f"Diff: {total_diff:.2f}, Percent Diff: {percent_diff:.2f}%")

        if percent_diff <= 20:
            print("✅ Within acceptable threshold.")
            break

        # Adjust total_count proportionally
        ratio = total_expected / total_simulated if total_simulated != 0 else 1.0
        total_count = int(total_count * ratio )
        total_count = max(1, total_count)  # prevent zero

        attempt += 1
