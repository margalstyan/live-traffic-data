import datetime
import os
import torch
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from simulation.generate_rou_single import generate_routes_for_next_timestamp
from model import SUMOGymEnv
from callbacks import GreenPhaseLoggerCallback

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

env = DummyVecEnv([make_env])  # wraps env for SB3 compatibility


# === TRAINING ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=1e-4,
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
#     "checkpoints_sb3/ppo_traffic_10240_steps.zip",
#     env=env,
#     device=device,
# )

env.envs[0].policy = model.policy  # Let env apply policy once per episode
model.set_logger(custom_logger)

# === TRAINING ===
callbacks = [checkpoint_callback, GreenPhaseLoggerCallback()]

model.learn(
    total_timesteps=100_000,
    callback=callbacks
)
