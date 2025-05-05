import os
import time
import numpy as np
import pandas as pd
from lxml import etree
import traci
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# === CONFIG ===
C = 300  # Capacity (vehicles/hour)
FLOW_DURATION = 300  # seconds
JUNCTION_IDS_TO_PROCESS = [3]
SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"
TIMESTAMP_COLUMN = "duration_20250327_1730"

# === BPR Volume Function ===
def compute_volume(t_f, t_obs, a, b):
    if t_obs <= t_f or a == 0:
        return 0
    return C * ((t_obs / t_f - 1) / a) ** (1 / b)

# === Generate SUMO Route File ===
def generate_flow_route_file_bpr(df, route_cache, params, route_file):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

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
        etree.SubElement(root, "flow", id=rid, type="car", route=rid, begin="0", end=str(FLOW_DURATION),
                         number=str(int_vehicle_count), departPos="random", arrivalPos="random")

    etree.ElementTree(root).write(route_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

# === Run SUMO Simulation ===
def run_sumo(route_file, tripinfo_file):
    if os.path.exists(tripinfo_file):
        os.remove(tripinfo_file)

    traci.start([
        SUMO_BINARY, "-c", SUMO_CONFIG, "-r", route_file,
        "--tripinfo-output", tripinfo_file, "--start", "--step-length", "1",
        "--random", "false", "--seed", "42"
    ])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

# === Read Trip Durations ===
def read_simulated_durations(tripinfo_file):
    tree = etree.parse(tripinfo_file)
    root = tree.getroot()
    durations_by_route = {}

    for trip in root.findall("tripinfo"):
        try:
            route_number = int(trip.get("id").split("_")[1].split(".")[0])
            durations_by_route.setdefault(route_number, []).append(float(trip.get("duration")))
        except Exception:
            continue

    return {k: np.mean(v) for k, v in durations_by_route.items()}

# === Build Route Cache ===
def build_route_cache(df_filtered):
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
    return route_cache

# === Multi-Objective Optimization Problem ===
class TrafficBPRProblem(Problem):
    def __init__(self, df, route_cache, n_routes):
        super().__init__(n_var=2 * n_routes,
                         n_obj=n_routes,
                         n_constr=0,
                         xl=np.array([0.01, 1.0] * n_routes),
                         xu=np.array([1.0, 8.0] * n_routes))
        self.df = df
        self.route_cache = route_cache
        self.n_routes = n_routes

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for params in X:
            try:
                run_id = "_".join(f"{x:.3f}" for x in params[:4]).replace(".", "_")
                route_file = f"generated_{run_id}.rou.xml"
                tripinfo_file = f"tripinfo_{run_id}.xml"
                generate_flow_route_file_bpr(self.df, self.route_cache, params, route_file)

                sim_results = {}
                for _ in range(10):  # Repeat for averaging
                    run_sumo(route_file, tripinfo_file)
                    results = read_simulated_durations(tripinfo_file)
                    for k, v in results.items():
                        sim_results[k] = sim_results.get(k, 0) + v
                for k in sim_results:
                    sim_results[k] /= 10
            except Exception as e:
                sim_results = {}

            df_filtered = self.df[self.df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
            route_errors = []
            for i, (idx, row) in enumerate(df_filtered.iterrows()):
                t_obs = row[TIMESTAMP_COLUMN]
                t_sim = sim_results.get(idx + 1, -1)
                if t_sim == -1:
                    route_errors.append(1e6)
                else:
                    route_errors.append((t_sim - t_obs) / t_obs)
            F.append(route_errors)
        out["F"] = np.array(F)

# === Run Optimizer ===
def run_multiobjective_optimization(df):
    df_filtered = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]
    n_routes = len(df_filtered)
    route_cache = build_route_cache(df_filtered)

    problem = TrafficBPRProblem(df, route_cache, n_routes)
    algorithm = NSGA2(pop_size=20)
    termination = get_termination("n_gen", 5)

    result = minimize(problem,
                      algorithm,
                      termination,
                      seed=1,
                      verbose=True)

    return result.X, result.F

# === Save Best Solution to CSV ===
def save_best_solution(X, F, output_path="best_solution.csv"):
    best_index = None
    best_score = float("inf")

    for i, errors in enumerate(F):
        avg_abs_error = np.mean(np.abs(errors))
        if avg_abs_error < best_score:
            best_score = avg_abs_error
            best_index = i

    best_params = X[best_index]
    best_errors = F[best_index]

    data = {
        "param_index": list(range(len(best_params))),
        "param_value": best_params,
        "route_error": [*best_errors, *([None] * (len(best_params) - len(best_errors)))]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Best solution saved to {output_path} with average absolute error: {best_score:.4f}")

# === Main Entry ===
if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("data/final_with_all_data.csv")
    X, F = run_multiobjective_optimization(df)
    print("✅ Optimization completed.")
    for i, (params, errors) in enumerate(zip(X, F)):
        print(f"\n--- Solution {i+1} ---")
        for j in range(len(errors)):
            print(f"Route {j+1} error: {errors[j]:.4f}")
        print("Params:", params)
    save_best_solution(X, F)
    print(f"\n⏱ Total time: {time.time() - start:.2f} seconds")
