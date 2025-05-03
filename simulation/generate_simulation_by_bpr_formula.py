import pandas as pd
from lxml import etree
import traci
from scipy.optimize import minimize
import numpy as np
import os
import time

# Constants
C = 1800  # Capacity (vehicles per hour)
FLOW_DURATION = 300  # seconds
JUNCTION_IDS_TO_PROCESS = [3]
SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"
TIMESTAMP_COLUMN = "duration_20250327_1730"
ROUTE_FILE = "generated_bpr.rou.xml"
TRIPINFO_FILE = "tripinfo.xml"


# Compute volume from observed time and free flow time using BPR formula
def compute_volume(t_f, t_obs, a, b):
    if t_obs <= t_f:
        return 0
    return C * ((t_obs / t_f - 1) / a) ** (1 / b)


# Generate the flow file using current a and b
def generate_flow_route_file_bpr(df, route_cache, a, b, route_file):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5",
                     length="5", maxSpeed="16.6", sigma="0.5")

    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)].copy()
    vehicle_counts = []

    for idx, row in df_filtered.iterrows():
        from_edge = row["from_edge"]
        to_edge = row["to_edge"]
        key = (from_edge, to_edge)

        if key not in route_cache:
            try:
                route_cache[key] = traci.simulation.findRoute(from_edge, to_edge).edges
            except Exception as e:
                print(f"Error finding route from {from_edge} to {to_edge}: {e}")
                route_cache[key] = []

    for idx, row in df_filtered.iterrows():
        rid = f"route_{idx+1}"
        from_edge = row["from_edge"]
        to_edge = row["to_edge"]
        t_f = row["duration_without_traffic"]
        t_obs = row[TIMESTAMP_COLUMN]

        vehicle_count = compute_volume(t_f, t_obs, a, b)
        vehicle_counts.append(vehicle_count)

        int_vehicle_count = int(vehicle_count * FLOW_DURATION / 3600)
        edges = route_cache.get((from_edge, to_edge), [])

        if not edges or int_vehicle_count == 0:
            continue

        etree.SubElement(root, "route", id=rid, edges=" ".join(edges))
        etree.SubElement(root, "flow", id=rid, type="car", route=rid, begin="0",
                         end=str(FLOW_DURATION), number=str(int_vehicle_count),
                         departPos="random", arrivalPos="random")

    # Save vehicle counts for calibration
    df.loc[df_filtered.index, "vehicle_count_estimate"] = vehicle_counts

    # Save route file
    tree = etree.ElementTree(root)
    tree.write(route_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")


# Run SUMO and get durations
def run_sumo():
    if os.path.exists(TRIPINFO_FILE):
        os.remove(TRIPINFO_FILE)

    traci.start([
        SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE,
        "--tripinfo-output", TRIPINFO_FILE,
        "--start", "--step-length", "1"
    ])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()


# Parse tripinfo.xml to get durations per route
def read_simulated_durations(tripinfo_file):
    tree = etree.parse(tripinfo_file)
    root = tree.getroot()
    durations_by_route = {}

    for trip in root.findall("tripinfo"):
        trip_id = trip.get("id")  # e.g., "route_37.1"
        if not trip_id:
            continue

        # Extract route index: from "route_37.1" => "37"
        try:
            route_number = int(trip_id.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue

        key = route_number
        duration = float(trip.get("duration"))

        if key not in durations_by_route:
            durations_by_route[key] = []

        durations_by_route[key].append(duration)

    # Average durations
    avg_durations = {key: np.mean(durations) for key, durations in durations_by_route.items()}
    return avg_durations



# The loss function to minimize
def loss(params, df, route_cache):
    a, b = params
    run_id = f"{a:.4f}_{b:.4f}".replace(".", "_")
    route_file = f"generated_{run_id}.rou.xml"
    tripinfo_file = f"tripinfo_{run_id}.xml"

    print(f"\nTesting alpha={a:.4f}, beta={b:.4f}")

    try:
        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start"])
        generate_flow_route_file_bpr(df, route_cache, a, b, route_file)
        traci.close()

        traci.start([
            SUMO_BINARY, "-c", SUMO_CONFIG, "-r", route_file,
            "--tripinfo-output", tripinfo_file,
            "--start", "--step-length", "1"
        ])
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
        traci.close()

        sim_results = read_simulated_durations(tripinfo_file)

    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e6  # Large penalty on failure

    # Filter df by routes we can compare
    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)].copy()

    errors = []
    for idx, row in df_filtered.iterrows():
        from_edge = row["from_edge"]
        to_edge = row["to_edge"]
        key = (from_edge, to_edge)
        t_obs = row[TIMESTAMP_COLUMN]

        if idx+1 in sim_results.keys():
            t_sim = sim_results[idx+1]
            error = (t_sim - t_obs) ** 2
            errors.append(error)
            print(f"Route {idx+1}: obs={t_obs:.2f}, sim={t_sim:.2f}, error={error:.2f}")

    if not errors:
        print("No matched routes to compute MSE.")
        return 1e6

    mse = np.mean(errors)
    print(f"✅ MSE: {mse:.4f}")
    return mse


# ========== Main Calibration Loop ==========
if __name__ == "__main__":
    df = pd.read_csv("data/final_with_all_data.csv")
    route_cache = {}

    start_time = time.time()

    result = minimize(
        fun=loss,
        x0=[0.15, 4],
        args=(df, route_cache),
        bounds=[(0.01, 2), (1.0, 10.0)],
        method="L-BFGS-B",
        options={"maxiter": 10}  # Increase for better accuracy
    )

    alpha_opt, beta_opt = result.x
    print(f"\n✅ Optimal alpha: {alpha_opt:.4f}, beta: {beta_opt:.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
