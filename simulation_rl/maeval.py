import os
import numpy as np
import pandas as pd
import torch
from lxml import etree
from stable_baselines3 import SAC
from simulation_rl.mamodel import MultiAgentRouteEnv
from simulation.generate_rou_single import JUSNTION_ID_TO_DATA_ID

# === Config ===
SUMO_CFG = "osm.sumocfg"
CSV_PATH = "../data/final_with_all_data.csv"
TIMESTAMP_COL = "duration_20250402_2015"
ROUTE_FILE = "eval_routes.rou.xml"
TRIPINFO_FILE = "eval_tripinfo.xml"
RUN_ID = "run10"
EPISODE_TO_LOAD = 700
MODEL_DIR = os.path.join("sac_agents", RUN_ID)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# === Load and prepare route mapping
def generate_tls_to_routes(csv_path: str, timestamp_column: str):
    df = pd.read_csv(csv_path)
    tls_to_routes = {}
    csv_id_to_tls = {v: k for k, v in JUSNTION_ID_TO_DATA_ID.items()}

    for i, row in df.iterrows():
        tls_id = csv_id_to_tls.get(row["Junction_id"])
        if not tls_id:
            continue
        route = {
            "id": f"r{i}",
            "from": row["from_edge"],
            "to": row["to_edge"],
            "distance": row["distance"],
            "duration_without_traffic": row["duration_without_traffic"],
            "target_duration": float(row[timestamp_column]),
            "csv_index": i
        }
        tls_to_routes.setdefault(tls_id, []).append(route)
    return tls_to_routes

tls_to_routes = generate_tls_to_routes(CSV_PATH, TIMESTAMP_COL)

# === Load Environment
env = MultiAgentRouteEnv(
    sumo_cfg=SUMO_CFG,
    tls_to_routes=tls_to_routes,
    csv_path=CSV_PATH,
    route_output_path=ROUTE_FILE,
    tripinfo_output_path=TRIPINFO_FILE,
    sumo_binary="sumo",
    max_vehicle_per_route=300,
    max_simulation_time=360
)
tls_ids = env.tls_ids

# === Load all SAC models
def load_agents():
    agents = {}
    for tls_id in tls_ids:
        model_path = os.path.join(MODEL_DIR, tls_id, f"ep_{EPISODE_TO_LOAD}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        agents[tls_id] = SAC.load(model_path, device=DEVICE)
    return agents

agents = load_agents()

# === Get initial observation and predict actions
obs, _ = env.reset()
actions = {
    tls_id: agents[tls_id].predict(obs[tls_id], deterministic=True)[0]
    for tls_id in tls_ids
}

# === Simulate
env.step(actions)

# === Parse tripinfo file directly
def parse_tripinfo_durations(tripinfo_path):
    route_durations = {}
    try:
        tree = etree.parse(tripinfo_path)
        for trip in tree.getroot().iter("tripinfo"):
            veh_id = trip.attrib["id"]
            route_id = veh_id.split('.')[0]
            duration = float(trip.attrib["duration"])
            route_durations.setdefault(route_id, []).append(duration)
    except Exception as e:
        print(f"❌ Error parsing tripinfo: {e}")
    return route_durations

actual_durations = parse_tripinfo_durations(TRIPINFO_FILE)

# === Evaluation results
print(f"\n📊 Evaluation Results for timestamp: {TIMESTAMP_COL}")
route_mse = []
total_abs_error = 0
total_target = 0
missing_routes = 0

for tls_id in tls_ids:
    for route in tls_to_routes[tls_id]:
        rid = f"route_{route['id']}"
        target = route["target_duration"]

        durs = actual_durations.get(rid, [])
        if not durs:
            print(f"⚠️ Missing duration for {rid}, fallback to target")
            pred = target
            missing_routes += 1
        else:
            pred = np.mean(durs)

        abs_diff = abs(pred - target)
        perc_diff = (abs_diff / target) * 100 if target else 0
        mse = (pred - target) ** 2

        route_mse.append(mse)
        total_abs_error += abs_diff
        total_target += target

        print(f"Route {rid}: Pred={pred:.2f}s | Target={target:.2f}s | Δ={abs_diff:.2f}s | %Δ={perc_diff:.2f}%")

# === Summary
network_mse = np.mean(route_mse)
perc_total = (total_abs_error / total_target) * 100 if total_target > 0 else 0

print(f"\n✅ MSE (Network): {network_mse:.2f}")
print(f"✅ Total Absolute Error: {total_abs_error:.2f}s")
print(f"✅ Total % Error: {perc_total:.2f}%")
print(f"✅ Missing route predictions: {missing_routes}")
