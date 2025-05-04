# multiagent_train_separate.py
import os
import numpy as np
from multiprocessing import Pool
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure

from simulation.generate_rou_single import generate_random_routes
from single_step_model import SUMOGymEnv
import traci

# Configuration
BASE_ROUTE_FILE = "xml/routes_ddpg_{agent_id}.rou.xml"
SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
LOG_DIR = "logs_multi_separate"
DEVICE = "cuda" if os.environ.get("USE_CUDA") else "cpu"

os.makedirs(LOG_DIR, exist_ok=True)

# Dynamically get TLS IDs

def get_tls_ids(sumo_config="osm.sumocfg", sumo_binary="sumo"):
    if traci.isLoaded():
        traci.close()
    traci.start([sumo_binary, "-c", sumo_config, "--start", "--step-length", "1"])
    tls_ids = traci.trafficlight.getIDList()
    traci.close()
    return tls_ids

TLS_IDS = get_tls_ids()

# Single training job for one agent
def train_agent(agent_id):
    print(f"\n🚦 Training agent for TLS: {agent_id}")

    route_file = BASE_ROUTE_FILE.format(agent_id=agent_id)
    generate_random_routes(output_file=route_file)

    env = SUMOGymEnv(
        sumo_config_path=SUMO_CONFIG,
        net_file_path=NET_FILE,
        tls_id=agent_id,
        use_gui=False,
        max_steps=300,
        route_file_path=route_file
    )

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    log_path = os.path.join(LOG_DIR, agent_id)
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["stdout", "tensorboard"])

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=lambda progress: 1e-3 * progress,
        action_noise=action_noise,
        learning_starts=1000,
        batch_size=256,
        verbose=1,
        device=DEVICE,
        tensorboard_log=log_path
    )

    model.set_logger(logger)

    model.learn(total_timesteps=5_000)
    model.save(os.path.join(log_path, f"ddpg_{agent_id}"))

    env.close()

# Parallel training loop using Pool
def train_all_agents_parallel():
    with Pool(processes=min(len(TLS_IDS), os.cpu_count())) as pool:
        pool.map(train_agent, TLS_IDS)

if __name__ == "__main__":
    train_all_agents_parallel()
