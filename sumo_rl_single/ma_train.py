import os
import torch
from torch.utils.tensorboard import SummaryWriter
from sac_agent import SACAgent
from buffer import ReplayBuffer
from mamodel import MultiSUMOGymEnv
from sumo_rl_single.maddpg import get_tls_ids

# === Config ===
TLS_IDS = get_tls_ids()
SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
ROUTE_FILE = "routes.rou.xml"
USE_GUI = False

EP_START = 0
MAX_EPISODES = 10000 + EP_START

BATCH_SIZE = 128
START_TRAINING_AFTER = 128
TRAIN_EVERY = 1
SAVE_EVERY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# === Directories ===
os.makedirs("sac_models", exist_ok=True)
os.makedirs("logs_sac", exist_ok=True)

# === Helper: Load agent weights for fine-tuning ===
import re

def load_agent_checkpoints(tls_id, agent, checkpoint_dir="sac_models", episode: int = None):
    try:
        if episode is None:
            # Auto-detect the latest available episode
            files = os.listdir(checkpoint_dir)
            pattern = re.compile(rf"{tls_id}_actor_ep(\d+)\.pth")
            ep_numbers = [int(pattern.search(f).group(1)) for f in files if pattern.search(f)]
            if not ep_numbers:
                raise FileNotFoundError("No checkpoint files found.")
            episode = max(ep_numbers)

        print(f"📦 Loading episode {episode} for {tls_id}...")

        actor_path = os.path.join(checkpoint_dir, f"{tls_id}_actor_ep{episode}.pth")
        q1_path = os.path.join(checkpoint_dir, f"{tls_id}_q1_ep{episode}.pth")
        q2_path = os.path.join(checkpoint_dir, f"{tls_id}_q2_ep{episode}.pth")
        alpha_path = os.path.join(checkpoint_dir, f"{tls_id}_log_alpha_ep{episode}.pth")

        agent.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        agent.q1.load_state_dict(torch.load(q1_path, map_location=DEVICE))
        agent.q2.load_state_dict(torch.load(q2_path, map_location=DEVICE))

        if os.path.exists(alpha_path):
            log_alpha_value = torch.load(alpha_path)
            agent.log_alpha = torch.tensor(log_alpha_value, dtype=torch.float32, requires_grad=True, device=DEVICE)

        print(f"✅ Loaded weights for {tls_id} at episode {episode}")
    except Exception as e:
        print(f"⚠️ No checkpoint found for {tls_id}. Starting from scratch. Reason: {e}")

# === Env ===
env = MultiSUMOGymEnv(
    sumo_config_path=SUMO_CONFIG,
    net_file_path=NET_FILE,
    tls_ids=TLS_IDS,
    route_file_path=ROUTE_FILE,
    use_gui=USE_GUI,
    max_steps=400
)

# === Initial dummy reset to get shapes
initial_obs, _ = env.reset()

# === Agent Setup
agents = {}
buffers = {}
writers = {}

for tls_id in TLS_IDS:
    obs_shape = initial_obs[tls_id].shape[0]
    act_shape = env.action_space[tls_id].shape[0]
    act_limit = float(env.action_space[tls_id].high[0])
    writers[tls_id] = SummaryWriter(log_dir=f"logs_sac/{tls_id}")
    buffers[tls_id] = ReplayBuffer(obs_shape, act_shape)

    agent = SACAgent(obs_shape, act_shape, act_limit, device=DEVICE, writer=writers[tls_id])
    # load_agent_checkpoints(tls_id, agent, episode=0)
    agents[tls_id] = agent

# === Training Loop
global_step = 0

for ep in range(EP_START, MAX_EPISODES + 1):
    obs, _ = env.reset()
    episode_reward = {k: 0.0 for k in TLS_IDS}

    # === One step = one full simulation run
    actions = {
        tls_id: agents[tls_id].select_action(obs[tls_id], deterministic=False).detach().cpu().numpy().flatten()
        for tls_id in TLS_IDS
    }

    next_obs, rewards, dones, _, _ = env.step(actions)

    for tls_id in TLS_IDS:
        buffers[tls_id].store(
            obs[tls_id],
            actions[tls_id],
            rewards[tls_id],
            next_obs[tls_id],
            dones[tls_id]
        )
        episode_reward[tls_id] += rewards[tls_id]
        writers[tls_id].add_scalar("reward", rewards[tls_id], global_step)

    obs = next_obs
    global_step += 1

    # === Train
    if global_step > START_TRAINING_AFTER and global_step % TRAIN_EVERY == 0:
        for tls_id in TLS_IDS:
            if buffers[tls_id].size >= BATCH_SIZE:
                agents[tls_id].update(buffers[tls_id], BATCH_SIZE)

    # === Save Checkpoint
    if ep % SAVE_EVERY == 0:
        for tls_id in TLS_IDS:
            torch.save(agents[tls_id].actor.state_dict(), f"sac_models/{tls_id}_actor_ep{ep}.pth")
            torch.save(agents[tls_id].q1.state_dict(), f"sac_models/{tls_id}_q1_ep{ep}.pth")
            torch.save(agents[tls_id].q2.state_dict(), f"sac_models/{tls_id}_q2_ep{ep}.pth")
            torch.save(agents[tls_id].log_alpha.detach().cpu().item(), f"sac_models/{tls_id}_log_alpha_ep{ep}.pth")

    print(f"EP {ep}: " + ", ".join(f"{k}={episode_reward[k]:.2f}" for k in TLS_IDS))

# === Final Save
for tls_id in TLS_IDS:
    torch.save(agents[tls_id].actor.state_dict(), f"sac_models/{tls_id}_actor.pth")
    torch.save(agents[tls_id].q1.state_dict(), f"sac_models/{tls_id}_q1.pth")
    torch.save(agents[tls_id].q2.state_dict(), f"sac_models/{tls_id}_q2.pth")
    torch.save(agents[tls_id].log_alpha.detach().cpu().item(), f"sac_models/{tls_id}_log_alpha.pth")

# === Close
env.close()
for tls_id in TLS_IDS:
    writers[tls_id].close()
    print(f"✅ Closed TensorBoard writer for {tls_id}")
print("✅ Training complete.")
