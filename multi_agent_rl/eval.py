import os
import numpy as np
import pandas as pd
import torch
from env import SUMOMultiAgentEnv
from stable_baselines3 import SAC

# === Config ===
SUMO_CFG = "osm.sumocfg"
ROUTE_CSV = "routes.csv"
USE_GUI = False  # Only if your env supports GUI toggle
RUN_ID = "run18"  # Change if using a subfolder, or leave "" for direct path
MODEL_DIR = os.path.join("sac_agents", RUN_ID) if RUN_ID else "sac_agents"
EVAL_EPISODES = 10
EPISODE_TO_LOAD = 6000  # Match with saved checkpoint like ep_100.zip

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# === Load Environment
env = SUMOMultiAgentEnv(SUMO_CFG, ROUTE_CSV, sumo_port=8888)  # Remove use_gui if not in your env class
tls_ids = env.tls_ids


# === Load agents from SB3 SAC checkpoints
def load_sac_agents():
    agents = {}
    for tls_id in tls_ids:
        model_path = os.path.join(MODEL_DIR, tls_id, f"ep_{EPISODE_TO_LOAD}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        model = SAC.load(model_path, device=DEVICE)
        agents[tls_id] = model
    return agents


# === Evaluate a single episode
def run_episode(env, agents=None, deterministic=True):
    obs, _ = env.reset()
    if agents:
        actions = {
            tls_id: agents[tls_id].predict(obs[tls_id], deterministic=deterministic)[0]
            for tls_id in tls_ids
        }
    else:
        actions = {
            tls_id: np.zeros(env.agent_action_spaces[tls_id].shape)
            for tls_id in tls_ids
        }

    _, _, _, _, infos = env.step(actions)

    wait_total, dur_total, count_total = 0, 0, 0
    for tls_id in tls_ids:
        info = infos.get(tls_id, [{}])[0]
        vc = info.get("vehicle_count", 0)
        wait_total += info.get("avg_wait", 0) * vc
        dur_total += info.get("avg_duration", 0) * vc
        count_total += vc

    wait_avg = wait_total / count_total if count_total else 0
    dur_avg = dur_total / count_total if count_total else 0
    return wait_avg, dur_avg


# === Run multiple episodes
def evaluate(env, agents=None, label="Model"):
    wait_list, dur_list = [], []
    for i in range(EVAL_EPISODES):
        wait, dur = run_episode(env, agents=agents)
        print(f"[{label}] Episode {i+1}: wait={wait:.2f}, duration={dur:.2f}")
        wait_list.append(wait)
        dur_list.append(dur)
    return np.mean(wait_list), np.mean(dur_list)


# === Run Evaluation
print("\n🔁 Evaluating SUMO static baseline...")
wait_static, dur_static = evaluate(env, agents=None, label="Static")

print("\n🤖 Evaluating trained MARL SAC model...")
agents = load_sac_agents()
wait_model, dur_model = evaluate(env, agents=agents, label="Model")

# === Print Comparison
wait_diff = wait_static - wait_model
dur_diff = dur_static - dur_model
wait_pct = (wait_diff / wait_static * 100) if wait_static else 0
dur_pct = (dur_diff / dur_static * 100) if dur_static else 0

df = pd.DataFrame({
    "Metric": ["Avg Waiting Time", "Avg Travel Duration"],
    "Static": [wait_static, dur_static],
    "Model": [wait_model, dur_model],
    "Improvement": [wait_diff, dur_diff],
    "Improvement (%)": [wait_pct, dur_pct]
})

print("\n📊 Evaluation Summary:")
print(df.to_string(index=False))

# Optional: Save to CSV
df.to_csv(f"eval_summary_{RUN_ID or 'default'}.csv", index=False)
