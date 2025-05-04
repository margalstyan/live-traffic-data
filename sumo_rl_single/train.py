import datetime
import os
import random
import numpy as np
import torch
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from simulation.generate_rou_single import generate_routes_for_next_timestamp
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
def make_env():
    generate_routes_for_next_timestamp()
    return SUMOGymEnv(
        sumo_config_path="osm.sumocfg",
        net_file_path="osm.net.xml",
        tls_id="cluster_2271368471_4779869278",
        use_gui=False,
        max_steps=300
    )

env = make_env()
env.seed(seed)


# === LOGGER ===
custom_logger = configure(log_dir, ["stdout", "tensorboard"])



# === PPO CONFIGURATION FOR FULL EPISODE TRAINING ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=16,               # 8 episodes per update
    batch_size=4,            # minibatch of 4 episodes
    n_epochs=10,             # repeat each batch 10 times
    learning_rate=1e-4,      # more stable updates
    clip_range=0.1,          # smaller, safer updates
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=1,
    tensorboard_log=log_dir,
    device=device
)
# model = PPO.load("./checkpoints_sb3/run_2025-05-03_01-48-50/ppo_traffic_744_steps.zip", env=env, device=device)
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
try:
    # === TRAINING LOOP ===
    model.learn(
        total_timesteps=500_000,
        callback=callbacks
    )
except Exception as e:
    print(f"⚠️ Training interrupted: {e}")
finally:
    env.close()
    if traci.isLoaded():
        traci.close()