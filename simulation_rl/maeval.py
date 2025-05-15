import os
import numpy as np
import pandas as pd
import torch
from env import SUMOMultiAgentEnv
from stable_baselines3 import SAC

# === Config ===
SUMO_CFG = "osm.sumocfg"
ROUTE_CSV = "routes.csv"
USE_GUI = False
RUN_ID = "run1"  # Change this to match your training run
MODEL_DIR = os.path.join("sac_agents", RUN_ID)
EVAL_EPISODES = 5
EPISODE_TO_LOAD = 1000  # Use the number from ep_1000.zip

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# === Load Environment
env = SUMOMultiAgentEnv(SUMO_CFG, ROUTE_CSV, sumo_port=8888)
tls_ids = env.tls_ids

# === Load SAC agents
def load_agents():
    agents = {}
    for tls_id in tls_ids:
        model_path = os.path.join(MODEL_DIR, tls_id, f"ep_{EPISODE_TO_LOAD}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        agent = SAC.load(model_path, device=DEVICE)
        agents[tls_id] = agent
    return agents

# === Evaluate one episode
def run_episode(env, agents=None, deterministic=True):
    obs, _ = env.reset()
    if agents:
        actions = {
            tls_id: agents[tls_id].predict(obs[tls_id], deterministic=deterministic)[0]
            for tls_id in tls_ids
        }
    else:
        actions = {
            tls_id: np.zeros(env.agent_action_spaces[tls_id].shape, dtype=np.float32)
            for tls_id in tls_ids
        }

    _, _, _, _, infos = env.step(actions)

    wait_total, dur_total, count_total = 0.0, 0.0, 0
    for tls_id in tls_ids:
        info = infos.get(tls_id, [{}])[0]
        count = info.get("vehicle_count", 0)
        wait_total += info.get("avg_wait", 0) * count
        dur_total += info.get("avg_duration", 0) * count
        count_total += count

    wait_avg = wait_total / count_total if count_total else 0.0
    dur_avg = dur_total / count_total if count_total else 0.0
    return wait_avg, dur_avg

# === Evaluate over multiple episodes
def evaluate(env, agents=None, label="Model"):
    waits, durs = [], []
    for i in range(EVAL_EPISODES):
        wait, dur = run_episode(env, agents=agents)
        print(f"[{label}] Episode {i+1}: avg_wait = {wait:.2f}, avg_duration = {dur:.2f}")
        waits.append(wait)
        durs.append(dur)
    return np.mean(waits), np.mean(durs)

# === Run evaluation
print("\n🔁 Evaluating static baseline...")
static_wait, static_dur = evaluate(env, agents=None, label="Static")

print("\n🤖 Evaluating trained SAC model...")
agents = load_agents()
model_wait, model_dur = evaluate(env, agents=agents, label="SAC")

# === Summary
wait_diff = static_wait - model_wait
dur_diff = static_dur - model_dur

df = pd.DataFrame({
    "Metric": ["Avg Waiting Time", "Avg Travel Duration"],
    "Static": [static_wait, static_dur],
    "Model": [model_wait, model_dur],
    "Improvement": [wait_diff, dur_diff],
    "Improvement (%)": [
        (wait_diff / static_wait * 100) if static_wait else 0,
        (dur_diff / static_dur * 100) if static_dur else 0
    ]
})

print("\n📊 Evaluation Summary:")
print(df.to_string(index=False))

# === Save
out_path = f"eval_summary_{RUN_ID}.csv"
df.to_csv(out_path, index=False)
print(f"\n💾 Saved summary to {out_path}")
