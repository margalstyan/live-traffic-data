import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
import gymnasium as gym

from simulation.generate_rou_single import JUSNTION_ID_TO_DATA_ID
from simulation_rl.mamodel import MultiAgentRouteEnv

# === Paths and Config ===
csv_path = "../data/final_with_all_data.csv"
sumo_cfg = "osm.sumocfg"
BASE_LOG_DIR = "./logs_marl"
BASE_MODEL_DIR = "./sac_agents"
TOTAL_EPISODES = 10000
SAVE_EVERY = 100
BATCH_SIZE = 100
TRAIN_AFTER_EP = 100

# === Auto-increment Run Directory ===
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

# === Generate tls_to_routes Mapping ===
def generate_tls_to_routes(csv_path: str) -> dict:
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
            "csv_index": i
        }
        tls_to_routes.setdefault(tls_id, []).append(route)
    return tls_to_routes

# === Dummy Env Wrapper ===
class DummySingleAgentEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space
    def reset(self, **kwargs): return self.observation_space.sample(), {}
    def step(self, action): return self.observation_space.sample(), 0.0, True, False, {}

# === Load Checkpoint or Create Agent ===
def load_or_create_agent(tls_id, dummy_env, logger, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, tls_id)
    if os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith(".zip")]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            model = SAC.load(os.path.join(checkpoint_path, latest), env=dummy_env)
            model.set_logger(logger)
            print(f"✅ Loaded checkpoint for {tls_id}: {latest}")
            return model
    # No checkpoint found, create new agent
    model = SAC(
        "MlpPolicy",
        dummy_env,
        verbose=0,
        buffer_size=100_000,
        learning_starts=100,
        batch_size=100,
        learning_rate=1e-3,
        tau=0.005,
        ent_coef="auto_0.3",
        tensorboard_log=logger.dir
    )
    model.set_logger(logger)
    return model

# === Load Data and Initialize Environment ===
tls_to_routes = generate_tls_to_routes(csv_path)
env = MultiAgentRouteEnv(
    sumo_cfg=sumo_cfg,
    tls_to_routes=tls_to_routes,
    csv_path=csv_path,
    route_output_path="routes_generated.rou.xml",
    tripinfo_output_path="tripinfo.xml",
    sumo_binary="sumo",
    max_simulation_time=400
)

# === Initialize Agents and Loggers ===
agents = {}
loggers = {}

for tls_id in tls_to_routes:
    dummy_env = DummySingleAgentEnv(env.observation_space[tls_id], env.action_space[tls_id])
    log_path = os.path.join(LOG_DIR, tls_id)
    checkpoint_dir = os.path.join(MODEL_DIR, tls_id)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = configure(log_path, ["stdout", "tensorboard"])
    loggers[tls_id] = logger
    agents[tls_id] = load_or_create_agent(tls_id, dummy_env, logger, MODEL_DIR)

print(f"🚦 Training {len(agents)} agents")
print(f"🧪 TensorBoard logs: {LOG_DIR}")
print(f"💾 Model checkpoints: {MODEL_DIR}")

# === Training Loop ===
for episode in range(1, TOTAL_EPISODES + 1):
    observations, _ = env.reset()
    actions = {
        tls_id: agents[tls_id].predict(observations[tls_id], deterministic=False)[0]
        for tls_id in tls_to_routes
    }

    next_obs, rewards, dones, _, infos = env.step(actions)

    for tls_id in tls_to_routes:
        agents[tls_id].replay_buffer.add(
            obs=observations[tls_id],
            action=actions[tls_id],
            reward=rewards[tls_id],
            next_obs=next_obs[tls_id],
            done=True,
            infos=[infos]
        )

        logger = loggers[tls_id]
        logger.record("train/reward", rewards[tls_id])

        if infos.get(tls_id) and isinstance(infos[tls_id], list):
            logger.record("env/avg_wait", infos[tls_id][0].get("avg_wait", 0))
            logger.record("env/vehicle_count", infos[tls_id][0].get("vehicle_count", 0))
            logger.record("env/avg_trip_duration", infos[tls_id][0].get("avg_duration", 0))

        logger.dump(step=episode)

        if episode >= TRAIN_AFTER_EP and agents[tls_id].replay_buffer.pos >= BATCH_SIZE:
            agents[tls_id].train(gradient_steps=1, batch_size=BATCH_SIZE)

    if episode % SAVE_EVERY == 0:
        print(f"[EP {episode}] 💾 Saving checkpoints...")
        for tls_id in tls_to_routes:
            model_path = os.path.join(MODEL_DIR, tls_id)
            os.makedirs(model_path, exist_ok=True)
            agents[tls_id].save(os.path.join(model_path, f"ep_{episode}.zip"))
