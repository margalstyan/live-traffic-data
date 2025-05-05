import os
import random
import numpy as np
import torch
import traci
import xml.etree.ElementTree as ET
import pandas as pd
from stable_baselines3 import DDPG,TD3, PPO, SAC
from single_step_model import SUMOGymEnv
from simulation.generate_rou_single import generate_random_routes
from multiprocessing import Pool, cpu_count

SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
CHECKPOINT_DIR = "checkpoints_multi_separate/DDPG2"
EVAL_ROUNDS = 10
MAX_STEPS = 600
CHECKPOINT_STEP = 1100
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_tls_ids():
    if traci.isLoaded():
        traci.close()
    traci.start(["sumo", "-c", SUMO_CONFIG, "--start", "--step-length", "1"])
    tls_ids = traci.trafficlight.getIDList()
    traci.close()
    return tls_ids

def read_tripinfo(path):
    tree = ET.parse(path)
    root = tree.getroot()
    durations = [float(t.attrib['duration']) for t in root.findall("tripinfo")]
    waiting_times = [float(t.attrib['waitingTime']) for t in root.findall("tripinfo")]
    return {
        "count": len(durations),
        "mean_duration": np.mean(durations) if durations else 0,
        "mean_waiting": np.mean(waiting_times) if waiting_times else 0
    }

def run_rl_model(tls_id, round_idx):
    route_file = f"xml/route_rl_{tls_id}_{round_idx}.rou.xml"
    tripinfo_out = f"xml/tripinfo_rl_{tls_id}_{round_idx}.xml"
    generate_random_routes(output_file=route_file, junction_id=tls_id)

    model_path = os.path.join(CHECKPOINT_DIR, tls_id, f"ddpg_{tls_id}_{CHECKPOINT_STEP}_steps")
    model = DDPG.load(model_path, device=DEVICE)

    env = SUMOGymEnv(
        sumo_config_path=SUMO_CONFIG,
        net_file_path=NET_FILE,
        tls_id=tls_id,
        use_gui=False,
        max_steps=MAX_STEPS,
        route_file_path=route_file
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action, tripinfo=tripinfo_out)
        done = terminated or truncated
    env.close()
    return tripinfo_out

def run_static_model(tls_id, round_idx):
    route_file = f"xml/route_static_{tls_id}_{round_idx}.rou.xml"
    tripinfo_out = f"xml/tripinfo_static_{tls_id}_{round_idx}.xml"
    generate_random_routes(output_file=route_file, junction_id=tls_id)

    sumo_binary = "sumo"
    traci.start([
        sumo_binary, "-c", SUMO_CONFIG,
        "--route-files", route_file,
        "--tripinfo-output", tripinfo_out
    ])
    step = 0
    while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    return tripinfo_out

def evaluate_tls(tls_id):
    print(f"🚦 Evaluating TLS: {tls_id}")
    rl_results, static_results = [], []

    for i in range(EVAL_ROUNDS):
        rl_path = run_rl_model(tls_id, i)
        static_path = run_static_model(tls_id, i)

        rl_stats = read_tripinfo(rl_path)
        static_stats = read_tripinfo(static_path)

        rl_results.append(rl_stats)
        static_results.append(static_stats)

    rl_mean = pd.DataFrame(rl_results).mean().to_dict()
    static_mean = pd.DataFrame(static_results).mean().to_dict()

    return {
        "tls_id": tls_id,
        "rl_duration": rl_mean['mean_duration'],
        "rl_waiting": rl_mean['mean_waiting'],
        "static_duration": static_mean['mean_duration'],
        "static_waiting": static_mean['mean_waiting'],
        "improvement_duration": static_mean['mean_duration'] - rl_mean['mean_duration'],
        "improvement_waiting": static_mean['mean_waiting'] - rl_mean['mean_waiting'],
        "percentage_improvement_duration": (
            (static_mean['mean_duration'] - rl_mean['mean_duration']) / static_mean['mean_duration'] * 100
            if static_mean['mean_duration'] > 0 else 0
        ),
        "percentage_improvement_waiting": (
            (static_mean['mean_waiting'] - rl_mean['mean_waiting']) / static_mean['mean_waiting'] * 100
            if static_mean['mean_waiting'] > 0 else 0
        )
    }

if __name__ == "__main__":
    TLS_IDS = get_tls_ids()

    # Use a process pool for parallel evaluation
    with Pool(processes=len(TLS_IDS)) as pool:
        summary_data = pool.map(evaluate_tls, TLS_IDS)

    df_summary = pd.DataFrame(summary_data)

    print("\n=== EVALUATION SUMMARY PER TLS ===")
    print(df_summary.to_string(index=False))

    print("\n=== OVERALL AVERAGES ===")
    print(df_summary.mean(numeric_only=True))

    df_summary.to_csv("evaluation_summary.csv", index=False)
    print("\n✅ Evaluation results saved to evaluation_summary.csv")
