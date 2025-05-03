import datetime
import os
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from simulation.generate_rou_single import generate_routes_for_next_timestamp
from single_step_model import SUMOGymEnv
from callbacks import GreenPhaseLoggerCallback
import random
import numpy as np
import torch

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
device = "cpu"

# === UNIQUE RUN NAME ===
run_name = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("logs_sb3", run_name)
checkpoint_dir = os.path.join("checkpoints_sb3", run_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"📝 Logging to:        {log_dir}")
print(f"💾 Checkpoints to:    {checkpoint_dir}")

# === ENVIRONMENT FACTORY ===
def make_env() -> Env:
    generate_routes_for_next_timestamp()
    return SUMOGymEnv(
        sumo_config_path="osm.sumocfg",
        net_file_path="osm.net.xml",
        tls_id="cluster_2271368471_4779869278",
        use_gui=False,
        max_steps=600
    )

env = make_env()
env.seed(seed)


# === TRAINING ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=16,
    ent_coef=0.01,
    clip_range=0.2,
    batch_size=64,
    learning_rate=3e-4,
    tensorboard_log=log_dir,
    device=device
)
custom_logger = configure(log_dir, ["stdout", "tensorboard"])

# === CHECKPOINT CALLBACK ===
checkpoint_callback = CheckpointCallback(
    save_freq=2048,
    save_path=checkpoint_dir,
    name_prefix="ppo_traffic"
)

# === LOAD MODEL OR TRAIN FROM SCRATCH ===
# model = PPO.load(
#     "./checkpoints_sb3/run_2025-05-02_21-06-10/ppo_traffic_36864_steps.zip",
#     env=env,
#     device=device,
# )
# env.policy = model.policy

model.set_logger(custom_logger)

# === TRAINING ===
callbacks = [checkpoint_callback, GreenPhaseLoggerCallback()]

model.learn(
    total_timesteps=500_000,
    callback=callbacks
)
