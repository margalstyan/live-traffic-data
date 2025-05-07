import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from model import MultiRouteSUMOGymEnv


def load_routes(csv_path: str, timestamp_column: str) -> list:
    df = pd.read_csv(csv_path)
    routes = []
    for i, row in df.iterrows():
        route = {
            "id": i,
            "edges": [str(row["from_edge"]), str(row["to_edge"])],
            "target_duration": float(row[timestamp_column]),
            "duration_without_traffic": float(row["duration_without_traffic"]),
            "distance": float(row["distance"])
        }
        routes.append(route)
    return routes


def evaluate_model(model_path: str, sumo_cfg: str, csv_path: str, timestamp: str,
                   route_output_file="predicted_routes.rou.xml",
                   tripinfo_file="tripinfo_eval.xml"):
    # === Load route data for the given timestamp ===
    routes = load_routes(csv_path, timestamp)

    # === Build env ===
    env = MultiRouteSUMOGymEnv(
        sumo_config_path=sumo_cfg,
        route_data_list=routes,
        route_file_path=route_output_file,
        tripinfo_path=tripinfo_file,
        sumo_gui=False,
        csv_path=csv_path
    )

    # === Prepare initial observation ===
    hhmm = timestamp.split('_')[-1]
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    minutes_of_day = hour * 60 + minute
    hour_norm = minutes_of_day / 1440.0

    obs = []
    for r in routes:
        dist = min(r["distance"] / env.max_distance, 1.0)
        free = min(r["duration_without_traffic"] / env.max_duration, 1.0)
        obs.append([dist, free, hour_norm])
    obs = np.array(obs, dtype=np.float32)

    # === Load model ===
    model = SAC.load(model_path)

    # === Predict actions ===
    actions, _ = model.predict(obs, deterministic=True)
    actions = np.clip(np.round(actions), 0, 300).astype(int)

    # === Generate route file ===
    routes_dict = {r["id"]: {"edges": r["edges"], "vehicle_count": actions[i]} for i, r in enumerate(routes)}
    env._generate_routes_file(routes_dict)

    # === Run simulation ===
    import traci
    if traci.isLoaded():
        traci.close()

    traci.start([
        env.sumo_binary,
        "-c", env.sumo_config,
        "--route-files", env.route_file_path,
        "--tripinfo-output", env.tripinfo_path,
        "--time-to-teleport", "300"
    ])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

    # === Parse tripinfo output ===
    actual_durations = {f"route_{r['id']}": [] for r in routes}
    try:
        tree = ET.parse(env.tripinfo_path)
        for trip in tree.getroot().findall("tripinfo"):
            rid = trip.attrib["id"].split('.')[0]
            duration = float(trip.attrib["duration"])
            if rid in actual_durations:
                actual_durations[rid].append(duration)
    except Exception as e:
        print(f"❌ Tripinfo parse error: {e}")
        return

    # === Calculate results ===
    print("\n📊 Evaluation Results for timestamp:", timestamp)
    route_mse = []
    total_absolute_error = 0
    total_target = 0

    for r in routes:
        rid = f"route_{r['id']}"
        target = r["target_duration"]
        pred_list = actual_durations.get(rid, [])
        if not pred_list:
            # print(f"⚠️  No vehicles arrived on {rid}")
            continue
        pred = np.mean(pred_list)
        abs_diff = abs(pred - target)
        perc_diff = (abs_diff / target) * 100 if target > 0 else 0
        mse = (pred - target) ** 2

        route_mse.append(mse)
        total_absolute_error += abs_diff
        total_target += target

        print(f"Route {rid}: Pred={pred:.2f}s | Target={target:.2f}s | "
              f"Δ={abs_diff:.2f}s | %Δ={perc_diff:.2f}%")

    network_mse = np.mean(route_mse)
    perc_total = (total_absolute_error / total_target) * 100 if total_target > 0 else 0

    print(f"\n✅ MSE (Network): {network_mse:.2f}")
    print(f"✅ Total Absolute Error: {total_absolute_error:.2f}s")
    print(f"✅ Total % Error: {perc_total:.2f}%")

    return {
        "mse": network_mse,
        "abs_diff_total": total_absolute_error,
        "perc_error": perc_total
    }


evaluate_model(
    model_path="./sac_checkpoints/1/sac1_model_1000_steps.zip",
    sumo_cfg="osm.sumocfg",
    csv_path="../data/final_with_all_data.csv",
    timestamp="duration_20250402_2015"
)
