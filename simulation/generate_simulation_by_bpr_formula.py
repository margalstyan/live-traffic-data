import pandas as pd
from lxml import etree
import traci
from scipy.optimize import differential_evolution
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

# Compute vehicle count from BPR formula
def compute_volume(t_f, t_obs, a, b):
    if t_obs <= t_f or a == 0:
        return 0
    return C * ((t_obs / t_f - 1) / a) ** (1 / b)

# Generate route XML file
def generate_flow_route_file_bpr(df, route_cache, a, b, route_file):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5",
                     length="5", maxSpeed="16.6", sigma="0.5")

    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)].copy()
    for idx, row in df_filtered.iterrows():
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

# Read simulated durations from tripinfo
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

# Loss function for optimization
def loss(params, df, route_cache):
    a, b = params
    run_id = f"{a:.4f}_{b:.4f}".replace(".", "_")
    route_file = f"generated_{run_id}.rou.xml"
    tripinfo_file = f"tripinfo_{run_id}.xml"

    print(f"\nTesting alpha={a:.4f}, beta={b:.4f}")
    try:
        generate_flow_route_file_bpr(df, route_cache, a, b, route_file)
        run_sumo(route_file, tripinfo_file)
        sim_results = read_simulated_durations(tripinfo_file)
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e6

    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)].copy()
    errors = []
    for idx, row in df_filtered.iterrows():
        t_obs = row[TIMESTAMP_COLUMN]
        if (idx + 1) in sim_results:
            t_sim = sim_results[idx + 1]
            error = (t_sim - t_obs) ** 2
            errors.append(error)
            print(f"Route {idx+1}: obs={t_obs:.2f}, sim={t_sim:.2f}, error={error:.2f}")

    if not errors:
        return 1e6
    mse = np.mean(errors)
    print(f"✅ MSE: {mse:.4f}")
    return mse

# Main block
if __name__ == "__main__":
    df = pd.read_csv("data/final_with_all_data.csv")
    route_cache = {}

    # Precompute all shortest paths
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start"])
    for _, row in df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)].iterrows():
        key = (row["from_edge"], row["to_edge"])
        if key not in route_cache:
            try:
                route_cache[key] = traci.simulation.findRoute(*key).edges
            except Exception as e:
                print(f"Route error for {key}: {e}")
                route_cache[key] = []
    traci.close()

    start_time = time.time()
    result = differential_evolution(
        func=loss,
        bounds=[(0.01, 1.0), (1.0, 8.0)],
        args=(df, route_cache),
        strategy="best1bin",
        maxiter=10,
        popsize=5,
        tol=1e-3,
        seed=42
    )

    print(f"\n✅ Optimal alpha: {result.x[0]:.4f}, beta: {result.x[1]:.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
