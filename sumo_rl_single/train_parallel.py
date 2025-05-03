import datetime
import os
import random
import numpy as np
import torch
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from simulation.generate_rou_single import generate_random_routes
from single_step_model import SUMOGymEnv
from callbacks import GreenPhaseLoggerCallback

# === Reproducibility ===
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# === DEVICE SELECTION ===
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ Using CUDA GPU")
else:
    device = "cpu"
    print("⚠️ Using CPU only")

# === RUN SETUP ===
run_name = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("logs_sb3", run_name)
checkpoint_dir = os.path.join("checkpoints_sb3", run_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"📝 Logging to:        {log_dir}")
print(f"💾 Checkpoints to:    {checkpoint_dir}")


# === ENVIRONMENT FACTORY ===
def make_env(rank):
    def _init():
        route_file = f"routes_env_{rank}.rou.xml"
        generate_random_routes(output_route_file=route_file)
        return SUMOGymEnv(
            sumo_config_path="osm.sumocfg",
            net_file_path="osm.net.xml",
            tls_id="cluster_2271368471_4779869278",
            use_gui=False,
            max_steps=300,
            route_file_path=route_file
        )

    return _init


if __name__ == "__main__":
    try:
        # === CREATE VECTORIZED ENVIRONMENTS ===
        num_envs = 8
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

        # === LOGGER ===
        custom_logger = configure(log_dir, ["stdout", "tensorboard"])

        # === PPO CONFIGURATION FOR PARALLEL EPISODE TRAINING ===
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=32,  # shared rollout across all envs
            batch_size=8,  # minibatch size
            n_epochs=20,
            learning_rate=3e-4,
            clip_range=0.1,
            ent_coef=0.3,
            vf_coef=1,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
        # model = PPO.load("./checkpoints_sb3/run_2025-05-03_19-09-42/ppo_traffic_3712_steps.zip", env=env, device=device, clip_range=0.2, learning_rate=1e-4, ent_coef=0.01, n_steps=64, batch_size=16, n_epochs=10)
        model.set_logger(custom_logger)

        # === CALLBACKS ===
        callbacks = [
            CheckpointCallback(
                save_freq=8,
                save_path=checkpoint_dir,
                name_prefix="ppo_traffic"
            ),
            GreenPhaseLoggerCallback(),
        ]
        model.learn(
            total_timesteps=100_000,
            callback=callbacks
        )
    except Exception as e:
        print(f"⚠️ Training interrupted: {e}")
    finally:
        if traci.isLoaded():
            traci.close()
