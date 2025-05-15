import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from env import SUMOMultiAgentEnv

# === Config ===
NUM_EPISODES = 10000
SAVE_EVERY = 100
BUFFER_SIZE = 50000
BATCH_SIZE = 128
TRAIN_AFTER_EP = 128

SUMO_CFG = "osm.sumocfg"
ROUTE_CSV = "routes.csv"
BASE_LOG_DIR = "./logs_marl"
BASE_MODEL_DIR = "./sac_agents"

# === Helper: Auto-increment run folder ===
def get_next_run_id(base_path):
    os.makedirs(base_path, exist_ok=True)
    existing = [d for d in os.listdir(base_path) if d.startswith("run")]
    existing_ids = [int(d[3:]) for d in existing if d[3:].isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    return f"run{next_id}"

RUN_ID = get_next_run_id(BASE_LOG_DIR)
LOG_DIR = os.path.join(BASE_LOG_DIR, RUN_ID)
MODEL_DIR = os.path.join(BASE_MODEL_DIR, RUN_ID)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Dummy Env for SAC agent init
class DummySingleAgentEnv(gym.Env):
    def __init__(self, obs_space: spaces.Box, act_space: spaces.Box):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

# === Initialize Environment
env = SUMOMultiAgentEnv(SUMO_CFG, ROUTE_CSV)
tls_ids = env.tls_ids

agents = {}
loggers = {}

for tls_id in tls_ids:
    obs_space = env.agent_observation_spaces[tls_id]
    act_space = env.agent_action_spaces[tls_id]
    dummy_env = DummySingleAgentEnv(obs_space, act_space)

    log_path = os.path.join(LOG_DIR, tls_id)
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["stdout", "tensorboard"])

    model = SAC(
        policy="MlpPolicy",
        env=dummy_env,
        learning_rate=3e-4,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        verbose=0,
        tensorboard_log=log_path
    )
    model.set_logger(logger)

    agents[tls_id] = model
    loggers[tls_id] = logger

print(f"🚦 Training {len(tls_ids)} agents in shared SUMO environment")
print(f"🧪 Logs: {LOG_DIR}")
print(f"💾 Models: {MODEL_DIR}")

# === Training Loop
for episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    actions = {
        tls_id: agents[tls_id].predict(obs[tls_id], deterministic=False)[0]
        for tls_id in tls_ids
    }
    next_obs, rewards, dones, _, infos = env.step(actions)

    for tls_id in tls_ids:
        # ✅ Add sample to SB3's internal replay buffer
        agents[tls_id].replay_buffer.add(
            obs=obs[tls_id],
            next_obs=next_obs[tls_id],
            action=actions[tls_id],
            reward=rewards[tls_id],
            done=dones[tls_id],
            infos=infos[tls_id],
        )

        # ✅ Log per-agent reward + env metrics
        logger = loggers[tls_id]
        logger.record("train/reward", rewards[tls_id])

        if infos[tls_id]:
            logger.record("env/avg_wait", infos[tls_id][0].get("avg_wait", 0))
            logger.record("env/vehicle_count", infos[tls_id][0].get("vehicle_count", 0))
            logger.record("env/avg_trip_duration", infos[tls_id][0].get("avg_duration", 0))

        logger.dump(step=episode)

        # ✅ Train only if buffer is populated enough
        if episode >= TRAIN_AFTER_EP and agents[tls_id].replay_buffer.pos >= BATCH_SIZE:
            agents[tls_id].train(gradient_steps=1, batch_size=BATCH_SIZE)

    # ✅ Save checkpoint every N episodes
    if episode % SAVE_EVERY == 0:
        print(f"[EP {episode}] 💾 Saving models...")
        for tls_id in tls_ids:
            tls_dir = os.path.join(MODEL_DIR, tls_id)
            os.makedirs(tls_dir, exist_ok=True)
            agents[tls_id].save(os.path.join(tls_dir, f"ep_{episode}.zip"))
