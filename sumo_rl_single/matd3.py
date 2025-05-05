import os
import numpy as np
from multiprocessing import Pool

import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from simulation.generate_rou_single import generate_random_routes
from single_step_model import SUMOGymEnv
import traci

# Configuration
BASE_ROUTE_FILE = "xml/routes_td3_{agent_id}.rou.xml"
SUMO_CONFIG = "osm.sumocfg"
NET_FILE = "osm.net.xml"
LOG_DIR = "logs_multi_separate/TD3"
CHECKPOINT_DIR = "checkpoints_multi_separate/TD3"

DEVICE =  "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Callback to log custom metrics
class GreenPhaseLoggerCallback(BaseCallback):
    def __init__(self, log_every=1):
        super().__init__()
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True

        infos = self.locals.get("infos", [])

        for info in infos:
            if "ep_length_real" in info:
                self.logger.record("custom/ep_length_real", info["ep_length_real"])

            if "green_durations" in info:
                durations = np.array(info["green_durations"])
                normalized = (durations - 10) / (90 - 10)
                for i, dur in enumerate(normalized):
                    self.logger.record(f"phases/green_phase_{i}", float(dur))

            if "ep_rew_mean" in info:
                self.logger.record("custom/ep_rew_mean", info["ep_rew_mean"])

        self.logger.dump(self.num_timesteps)
        return True

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
    generate_random_routes(output_file=route_file, junction_id=agent_id)

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

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=lambda progress: 1e-3 * progress,
        action_noise=action_noise,
        learning_starts=256,
        batch_size=256,
        verbose=1,
        device=DEVICE,
        tensorboard_log=log_path

    )
    # LAST_STEP = 400
    # model = DDPG2.load(f"{CHECKPOINT_DIR}/{agent_id}/ddpg_{agent_id}_{LAST_STEP}_steps.zip", env=env, device=DEVICE, learning_rate=lambda progress: 9e-3 * progress, ent_coef=0.02, n_steps=256, batch_size=256, n_epochs=20)
    model.set_logger(logger)

    callbacks = [
        CheckpointCallback(
            save_freq=100,
            save_path=os.path.join(CHECKPOINT_DIR, agent_id),
            name_prefix=f"td3_{agent_id}"
        ),
        GreenPhaseLoggerCallback(),
    ]

    model.learn(total_timesteps=5_000, callback=callbacks)
    model.save(os.path.join(log_path, f"td3_{agent_id}"))

    env.close()

# Parallel training loop using Pool
def train_all_agents_parallel():
    with Pool(processes=len(TLS_IDS)) as pool:
        pool.map(train_agent, TLS_IDS)

if __name__ == "__main__":
    train_all_agents_parallel()
