import os
import random

import numpy as np
import torch
import traci
import xml.etree.ElementTree as ET
from stable_baselines3 import DDPG
from single_step_model import SUMOGymEnv


SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
TLS_ID = "cluster_2271368471_4779869278"
ROUTE_GEN = True  # set True if you want to call generate_routes_for_next_timestamp()
MAX_STEPS = 600
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# === STEP 1: RL-CONTROLLED RUN ===
def run_rl_model(model_path: str, tripinfo_out="tripinfo_rl.xml", use_gui=False):
    if ROUTE_GEN:
        from simulation.generate_rou_single import generate_random_routes
        generate_random_routes(output_route_file="../sumo_rl_single/routes.rou.xml")

    env = SUMOGymEnv(
        sumo_config_path=SUMO_CONFIG,
        net_file_path=NET_FILE,
        tls_id=TLS_ID,
        use_gui=use_gui,
        max_steps=MAX_STEPS
    )
    env.seed(seed)

    model = DDPG.load(model_path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action, tripinfo=tripinfo_out)
        done = terminated or truncated
    env.close()

    # os.rename("tripinfo.xml", tripinfo_out)


# === STEP 2: STATIC BASELINE RUN ===
def run_static(tripinfo_out="tripinfo_static.xml", use_gui=False):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    traci.start([
        sumo_binary, "-c", SUMO_CONFIG,
        "--tripinfo-output", tripinfo_out
    ])
    step = 0
    while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()

    # os.rename("tripinfo.xml", tripinfo_out)


# === STEP 3: Tripinfo Parser ===
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


# === STEP 4: Compare Both Runs ===
if __name__ == "__main__":
    model_path = "./checkpoints_sb3/run_2025-05-03_21-43-45/ddpg_traffic_1664_steps.zip"

    print("🎯 Running RL-controlled simulation...")
    run_rl_model(model_path)

    print("📊 Running static traffic light baseline...")
    run_static()

    rl_stats = read_tripinfo("tripinfo_rl.xml")
    static_stats = read_tripinfo("tripinfo_static.xml")

    print("\n====== Evaluation Summary ======")
    print(f"Vehicles processed: {rl_stats['count']}")
    print(f"\n🔬 RL-Controlled:")
    print(f"  Avg Duration:     {rl_stats['mean_duration']:.2f} sec")
    print(f"  Avg Waiting Time: {rl_stats['mean_waiting']:.2f} sec")

    print(f"\n🧱 Static Baseline:")
    print(f"  Avg Duration:     {static_stats['mean_duration']:.2f} sec")
    print(f"  Avg Waiting Time: {static_stats['mean_waiting']:.2f} sec")

    print(f"\n🏁 Improvements by RL:")
    print(f"  Travel Time:      {static_stats['mean_duration'] - rl_stats['mean_duration']:.2f} sec")
    print(f"  Waiting Time:     {static_stats['mean_waiting'] - rl_stats['mean_waiting']:.2f} sec")
