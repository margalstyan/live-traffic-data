import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from vmodel import MultiRouteSUMOGymEnv  # ✅ Your custom env

# === Minimal dummy setup ===
csv_path = "../data/final_with_all_data.csv"
sumo_config = "osm.sumocfg"
checkpoint_path = "./vcheckpoints/2/vsac_6000_steps.zip"

# === Load route data (simplified for test) ===
import pandas as pd
df = pd.read_csv(csv_path)
route_data = [{
    "id": i,
    "edges": [str(row["from_edge"]), str(row["to_edge"])],
    "duration_without_traffic": float(row["duration_without_traffic"]),
    "distance": float(row["distance"])
} for i, row in df.iterrows()]

# === Create one test env
def make_env():
    return MultiRouteSUMOGymEnv(
        sumo_config_path=sumo_config,
        route_data_list=route_data,
        route_file_path="routes_0.rou.xml",
        tripinfo_path="tripinfo_0.xml",
        sumo_gui=False,
        csv_path=csv_path,
        instance_id=0
    )

env = DummyVecEnv([make_env])

# === Create model with updated settings
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=0,
    batch_size=2048,
    learning_starts=100,
    gradient_steps=-1,
    learning_rate=3e-4,
    train_freq=(1, "step"),
    buffer_size=1_000_000,
    action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0])),
    policy_kwargs={"net_arch": [256, 256]}
)

# === Reset env and get a dummy observation
obs = env.reset()

# === Predict action BEFORE loading checkpoint
action_before, _ = model.predict(obs, deterministic=True)

# === Print some weights before
print("🔎 Actor weights before loading:")
print(model.actor.mu.weight.data.flatten()[:5])

# === Load weights
model.set_parameters(checkpoint_path)

# === Predict action AFTER loading checkpoint
action_after, _ = model.predict(obs, deterministic=True)

# === Print some weights after
print("\n✅ Actor weights after loading:")
print(model.actor.mu.weight.data.flatten()[:5])

# === Compare actions
print("\n🎯 Action before loading weights:", action_before)
print("🎯 Action after loading weights: ", action_after)

# === Simple conclusion
if not np.allclose(action_before, action_after):
    print("\n✅ Checkpoint weights were successfully loaded and affected model behavior.")
else:
    print("\n⚠️ No difference detected. Check if the checkpoint was correct or if model architectures match.")

