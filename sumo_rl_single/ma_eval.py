import os
import torch
import numpy as np
import pandas as pd
import traci

from sac_agent import SACAgent
from mamodel import MultiSUMOGymEnv
from sumo_rl_single.maddpg import get_tls_ids

# === Config
TLS_IDS = get_tls_ids()
SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
ROUTE_FILE = "eval_routes.rou.xml"  # use a clean route file for evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPISODE_TO_LOAD = 500

# === Load Environment
env = MultiSUMOGymEnv(
    sumo_config_path=SUMO_CONFIG,
    net_file_path=NET_FILE,
    tls_ids=TLS_IDS,
    route_file_path=ROUTE_FILE,
    use_gui=False,
    max_steps=600  # same as training
)

# === Get obs/action dimensions
initial_obs, _ = env.reset()
obs_dims = {tls_id: initial_obs[tls_id].shape[0] for tls_id in TLS_IDS}
act_dims = {tls_id: env.action_space[tls_id].shape[0] for tls_id in TLS_IDS}
act_limits = {tls_id: float(env.action_space[tls_id].high[0]) for tls_id in TLS_IDS}

# === Load agents
def load_agent(tls_id, obs_shape, act_shape, act_limit, episode):
    agent = SACAgent(obs_shape, act_shape, act_limit, device=DEVICE)
    base = f"sac_models/{tls_id}"
    agent.actor.load_state_dict(torch.load(f"{base}_actor_ep{episode}.pth", map_location=DEVICE))
    agent.q1.load_state_dict(torch.load(f"{base}_q1_ep{episode}.pth", map_location=DEVICE))
    agent.q2.load_state_dict(torch.load(f"{base}_q2_ep{episode}.pth", map_location=DEVICE))

    alpha_path = f"{base}_log_alpha_ep{episode}.pth"
    if os.path.exists(alpha_path):
        alpha_val = torch.load(alpha_path)
        agent.log_alpha = torch.tensor(alpha_val, dtype=torch.float32, requires_grad=True, device=DEVICE)

    return agent

agents = {
    tls_id: load_agent(tls_id, obs_dims[tls_id], act_dims[tls_id], act_limits[tls_id], EPISODE_TO_LOAD)
    for tls_id in TLS_IDS
}

# === Evaluation Function
def evaluate(env, agents=None, use_model=True, iterations=10):
    wait_list, dur_list = [], []
    for _ in range(iterations):
        obs, _ = env.reset()

        if use_model:
            actions = {
                tls_id: agents[tls_id].select_action(obs[tls_id], deterministic=True).detach().cpu().numpy()
                for tls_id in TLS_IDS
            }
        else:
            actions = {
                tls_id: np.zeros(env.action_space[tls_id].shape)
                for tls_id in TLS_IDS
            }

        _, _, _, _, infos = env.step(actions)
        if not traci.isLoaded():
            traci.start([env.sumo_binary, "-c", env.sumo_config_path, "--no-step-log", "true", "--step-length", "5"])
        stats = env._parse_tripinfo_per_tls(env.tripinfo_path)
        if traci.isLoaded():
            traci.close()
        wait_sum = sum(s["wait_sum"] for s in stats.values())
        dur_sum = sum(s["dur_sum"] for s in stats.values())
        count = sum(s["count"] for s in stats.values())

        wait_avg = wait_sum / count if count else 0
        dur_avg = dur_sum / count if count else 0
        wait_list.append(wait_avg)
        dur_list.append(dur_avg)

    return np.mean(wait_list), np.mean(dur_list)

# === Evaluate
wait_static, dur_static = evaluate(env, use_model=False)
wait_model, dur_model = evaluate(env, agents=agents, use_model=True)

# === Compare Results
wait_diff = wait_static - wait_model
dur_diff = dur_static - dur_model
wait_pct = (wait_diff / wait_static * 100) if wait_static else 0
dur_pct = (dur_diff / dur_static * 100) if dur_static else 0

# === Show Summary
df = pd.DataFrame({
    "Metric": ["Avg Waiting Time", "Avg Travel Duration"],
    "Static": [wait_static, dur_static],
    "Model": [wait_model, dur_model],
    "Improvement": [wait_diff, dur_diff],
    "Improvement (%)": [wait_pct, dur_pct]
})

print(df.to_string(index=False))
