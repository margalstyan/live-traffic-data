import os
import shutil
import torch
import numpy as np
from multiprocessing import Pool, set_start_method
from torch.utils.tensorboard import SummaryWriter
from sac_agent import SACAgent
from buffer import ReplayBuffer
from mamodel import MultiSUMOGymEnv
from sumo_rl_single.maddpg import get_tls_ids

# === CONFIG ===
NUM_ENVS = 4
MAX_EPISODES = 5000
MAX_STEPS = 1
BATCH_SIZE = 64
START_TRAINING_AFTER = 10
TRAIN_EVERY = 50
SAVE_EVERY = 50

TLS_IDS = get_tls_ids()
SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
ROUTE_FILE = "routes.rou.xml"
DEVICE = "cpu"

os.makedirs("sac_models", exist_ok=True)
os.makedirs("logs_sac_multi_env", exist_ok=True)

# === Prepare per-env route and tripinfo files ===
ENV_ROUTE_FILES = []
ENV_TRIPINFO_FILES = []
for i in range(NUM_ENVS):
    route_copy = f"routes_env{i}.rou.xml"
    tripinfo_out = f"tripinfo_env{i}.xml"
    shutil.copy(ROUTE_FILE, route_copy)
    ENV_ROUTE_FILES.append(route_copy)
    ENV_TRIPINFO_FILES.append(tripinfo_out)

# === AGENTS, BUFFERS, WRITERS ===
agents = {}
buffers = {}
writers = {}

env0 = MultiSUMOGymEnv(SUMO_CONFIG, NET_FILE, TLS_IDS, route_file_path=ENV_ROUTE_FILES[0], use_gui=False, tripinfo_path=ENV_TRIPINFO_FILES[0])
initial_obs, _ = env0.reset()
for tls_id in TLS_IDS:
    obs_dim = initial_obs[tls_id].shape[0]
    act_dim = env0.action_space[tls_id].shape[0]
    agents[tls_id] = SACAgent(obs_dim, act_dim, 1.0, device=DEVICE)
    buffers[tls_id] = ReplayBuffer(obs_dim, act_dim)
    writers[tls_id] = SummaryWriter(log_dir=f"logs_sac_multi_env/{tls_id}")
env0.close()

def get_agent_state_dicts():
    return {
        tls_id: {
            'actor': agents[tls_id].actor.state_dict(),
            'q1': agents[tls_id].q1.state_dict(),
            'q2': agents[tls_id].q2.state_dict()
        }
        for tls_id in TLS_IDS
    }

def run_env_instance(env_idx, agent_state_dicts):
    env = MultiSUMOGymEnv(
        sumo_config_path=SUMO_CONFIG,
        net_file_path=NET_FILE,
        tls_ids=TLS_IDS,
        route_file_path=ENV_ROUTE_FILES[env_idx],
        use_gui=False,
        max_steps=300,
        tripinfo_path=ENV_TRIPINFO_FILES[env_idx]
    )

    local_agents = {}
    for tls_id in TLS_IDS:
        obs_shape = env.observation_space[tls_id].shape[0]
        act_shape = env.action_space[tls_id].shape[0]
        act_limit = float(env.action_space[tls_id].high[0])
        agent = SACAgent(obs_shape, act_shape, act_limit, device="cpu")
        agent.actor.load_state_dict(agent_state_dicts[tls_id]['actor'])
        agent.q1.load_state_dict(agent_state_dicts[tls_id]['q1'])
        agent.q2.load_state_dict(agent_state_dicts[tls_id]['q2'])
        local_agents[tls_id] = agent

    obs, _ = env.reset()
    actions = {
        tls_id: local_agents[tls_id].select_action(obs[tls_id], deterministic=False)
        .detach().cpu().numpy().flatten()
        for tls_id in TLS_IDS
    }

    next_obs, rewards, dones, _, _ = env.step(actions)
    result = {tls_id: (obs[tls_id], actions[tls_id], rewards[tls_id], next_obs[tls_id], dones[tls_id]) for tls_id in TLS_IDS}
    env.close()
    return result

# === TRAINING ===
set_start_method("spawn", force=True)
if __name__ == "__main__":
    global_step = 0

    for ep in range(1, MAX_EPISODES + 1):
        print(f"🌍 Episode {ep}")
        agent_state_dicts = get_agent_state_dicts()

        with Pool(processes=NUM_ENVS) as pool:
            results = pool.starmap(run_env_instance, [(i, agent_state_dicts) for i in range(NUM_ENVS)])

        # === Aggregate and Log Rewards per Agent ===
        episode_rewards = {tls_id: 0.0 for tls_id in TLS_IDS}

        for result in results:
            for tls_id in TLS_IDS:
                obs, action, reward, next_obs, done = result[tls_id]
                buffers[tls_id].store(obs, action, reward, next_obs, done)
                episode_rewards[tls_id] += reward  # Sum over environments

        # Log episode rewards to TensorBoard
        for tls_id in TLS_IDS:
            writers[tls_id].add_scalar("Episode_Reward", episode_rewards[tls_id], ep)
            writers[tls_id].flush()

        if global_step > START_TRAINING_AFTER and global_step % TRAIN_EVERY == 0:
            for tls_id in TLS_IDS:
                if buffers[tls_id].size >= BATCH_SIZE:
                    agents[tls_id].update(buffers[tls_id], BATCH_SIZE)

        if ep % SAVE_EVERY == 0:
            for tls_id in TLS_IDS:
                torch.save(agents[tls_id].actor.state_dict(), f"sac_models/{tls_id}_actor_ep{ep}.pth")
                torch.save(agents[tls_id].q1.state_dict(), f"sac_models/{tls_id}_q1_ep{ep}.pth")
                torch.save(agents[tls_id].q2.state_dict(), f"sac_models/{tls_id}_q2_ep{ep}.pth")

        global_step += 1

    # === Final Save ===
    for tls_id in TLS_IDS:
        torch.save(agents[tls_id].actor.state_dict(), f"sac_models/{tls_id}_actor.pth")
        torch.save(agents[tls_id].q1.state_dict(), f"sac_models/{tls_id}_q1.pth")
        torch.save(agents[tls_id].q2.state_dict(), f"sac_models/{tls_id}_q2.pth")

    print("✅ Finished parallel training.")
