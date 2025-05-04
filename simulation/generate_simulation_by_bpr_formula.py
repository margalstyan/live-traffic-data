import pandas as pd
from lxml import etree
import traci
from scipy.optimize import differential_evolution
import numpy as np
import os
import time

# Constants
C = 900  # Capacity (vehicles per hour)
FLOW_DURATION = 300  # seconds
JUNCTION_IDS_TO_PROCESS = [3]
SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"
TIMESTAMP_COLUMN = "duration_20250327_1730"
N_REPEATS = 44  # number of times to average simulations

# Compute vehicle count using BPR formula
def compute_volume(t_f, t_obs, a, b):
    if t_obs <= t_f or a == 0:
        return 0
    return C * ((t_obs / t_f - 1) / a) ** (1 / b)

# Generate SUMO route file with vehicle flows
def generate_flow_route_file_bpr(df, route_cache, params, route_file):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5",
                     length="5", maxSpeed="16.6", sigma="0.5")

    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
    for i, (idx, row) in enumerate(df_filtered.iterrows()):
        a, b = params[2 * i], params[2 * i + 1]
        rid = f"route_{idx+1}"
        from_edge, to_edge = row["from_edge"], row["to_edge"]
        t_f, t_obs = row["duration_without_traffic"], row[TIMESTAMP_COLUMN]
        vehicle_count = compute_volume(t_f, t_obs, a, b)
        int_vehicle_count = int(vehicle_count * FLOW_DURATION / 3600)

        edges = route_cache.get((from_edge, to_edge), [])
        if not edges or int_vehicle_count == 0:
            continue

        etree.SubElement(root, "route", id=rid, edges=" ".join(edges))
        etree.SubElement(root, "flow", id=rid, type="car", route=rid, begin="0",
                         end=str(FLOW_DURATION), number=str(int_vehicle_count),
                         departPos="random", arrivalPos="random")

    etree.ElementTree(root).write(route_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

# Run SUMO simulation
def run_sumo(route_file, tripinfo_file):
    if os.path.exists(tripinfo_file):
        os.remove(tripinfo_file)

    traci.start([
        SUMO_BINARY, "-c", SUMO_CONFIG, "-r", route_file,
        "--tripinfo-output", tripinfo_file,
        "--start", "--step-length", "1"
    ])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

# Read average trip durations by route
def read_simulated_durations(tripinfo_file):
    tree = etree.parse(tripinfo_file)
    root = tree.getroot()
    durations_by_route = {}

    for trip in root.findall("tripinfo"):
        trip_id = trip.get("id")
        try:
            route_number = int(trip_id.split("_")[1].split(".")[0])
        except Exception:
            continue
        durations_by_route.setdefault(route_number, []).append(float(trip.get("duration")))

    return {k: np.mean(v) for k, v in durations_by_route.items()}

# Loss function
def loss(params, df, route_cache):
    run_id = "_".join(f"{x:.3f}" for x in params[:4]).replace(".", "_")
    route_file = f"generated_{run_id}.rou.xml"
    tripinfo_file = f"tripinfo_{run_id}.xml"

    try:
        generate_flow_route_file_bpr(df, route_cache, params, route_file)
        sim_results = {}

        for _ in range(N_REPEATS):
            run_sumo(route_file, tripinfo_file)
            results = read_simulated_durations(tripinfo_file)
            for k, v in results.items():
                sim_results[k] = sim_results.get(k, 0) + v

        for k in sim_results:
            sim_results[k] /= N_REPEATS
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e6

    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
    errors = []
    for i, (idx, row) in enumerate(df_filtered.iterrows()):
        t_obs = row[TIMESTAMP_COLUMN]
        sim_duration = sim_results.get(idx + 1)
        if sim_duration is not None:
            error = (sim_duration - t_obs) ** 2
            errors.append(error)

    mse = np.mean(errors) if errors else 1e6
    print("MSE:", mse)
    return mse

# Main
if __name__ == "__main__":
    df = pd.read_csv("data/final_with_all_data.csv")
    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
    n_routes = len(df_filtered)
    route_cache = {}

    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start"])
    for _, row in df_filtered.iterrows():
        key = (row["from_edge"], row["to_edge"])
        if key not in route_cache:
            try:
                route_cache[key] = traci.simulation.findRoute(*key).edges
            except Exception:
                route_cache[key] = []
    traci.close()

    bounds = [(0.01, 1.0), (1.0, 8.0)] * n_routes
    start_time = time.time()

    result = differential_evolution(
        func=loss,
        bounds=bounds,
        args=(df, route_cache),
        strategy="best1bin",
        maxiter=10,
        popsize=5,
        tol=1e-3,
        seed=42
    )

    best_params = result.x
    for i in range(n_routes):
        print(f"Route {i+1}: α={best_params[2*i]:.4f}, β={best_params[2*i+1]:.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
