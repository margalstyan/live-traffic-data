import pandas as pd
import numpy as np
import os
import traci

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList

from tensorboard_callback import TensorboardCallback
from tensorboard_callback import EarlyStoppingCallback
from model import SumoTrafficEnv

# --- CONFIGURATION ---
ROUTE_CSV_PATH = "../data/final_with_all_data.csv"
SUMO_CONFIG_PATH = "../config/osm.sumocfg"
TIMESTAMP_TO_USE = "duration_20250327_1740"
MAX_VEHICLES = 200
SUMO_BINARY = "sumo"

# --- Load data ---
df_full = pd.read_csv(ROUTE_CSV_PATH)
JUNCTION_IDS_TO_PROCESS = ["4", "9", "8", "5"]
df = df_full[df_full["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]

routes = {}
for idx, row in df.iterrows():
    route_id = f"route_{idx}"
    routes[route_id] = {
        "origin": row["Origin"],
        "destination": row["Destination"],
        "from_edge": row["from_edge"],
        "to_edge": row["to_edge"],
        "target_duration": row[TIMESTAMP_TO_USE],
        "duration_without_traffic": row["duration_without_traffic"]
    }

print(f"✅ {len(routes)} valid routes loaded after filtering.")

# --- Create Environment ---
env = SumoTrafficEnv(routes, sumo_config=SUMO_CONFIG_PATH, max_vehicles=MAX_VEHICLES)

# (Optional) Check environment
check_env(env, warn=True)

# --- Setup TD3 model ---
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./sumo_rl_tensorboard/",
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=2000,
    batch_size=128,
    train_freq=1,
    gradient_steps=1,
)

# --- Setup Callbacks ---
tensorboard_callback = TensorboardCallback(verbose=1)
early_stopping_callback = EarlyStoppingCallback(required_success_rate=0.8, tolerance=0.2, verbose=1)

callback = CallbackList([tensorboard_callback, early_stopping_callback])

# --- Train the model ---
TOTAL_TIMESTEPS = 100_000

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

os.makedirs("models", exist_ok=True)

try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
except KeyboardInterrupt:
    print("\n⛔ Training interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\n❗ Error during training: {e}")
finally:
    model.save("models/td3_sumo_traffic")
    print("\n✅ Model saved after training stopped!")