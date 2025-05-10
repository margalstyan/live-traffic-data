import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.vec_env import SubprocVecEnv

from vmodel import MultiRouteSUMOGymEnv  # Make sure this supports instance_id


def load_routes(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    routes = []
    for i, row in df.iterrows():
        route = {
            "id": i,
            "edges": [str(row["from_edge"]), str(row["to_edge"])],
            "duration_without_traffic": float(row["duration_without_traffic"]),
            "distance": float(row["distance"])
        }
        routes.append(route)
    return routes


# ✅ Vectorized environment maker
def make_env(instance_id, sumo_config, route_data, csv_path):
    def _init():
        return MultiRouteSUMOGymEnv(
            sumo_config_path=sumo_config,
            route_data_list=route_data,
            route_file_path=f"routes_{instance_id}.rou.xml",
            tripinfo_path=f"tripinfo_{instance_id}.xml",
            sumo_gui=False,
            csv_path=csv_path,
            instance_id=instance_id
        )
    return _init


# ✅ Updated logger for vectorized env
class TensorboardRewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.rewards = []  # stores all finished episode rewards

    def _on_step(self) -> bool:
        # `dones` and `rewards` are lists (one per env)
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")

        if dones is not None and rewards is not None:
            for done, reward in zip(dones, rewards):
                if done:
                    self.rewards.append(reward)
                    # Log raw reward
                    self.logger.record("rollout/episode_reward", reward)
                    # Fixed 1-step episode
                    self.logger.record("rollout/ep_len_mean", 1.0)
                    # Moving average
                    self.logger.record("rollout/ep_rew_mean", np.mean(self.rewards[-100:]))

        return True
if __name__ == "__main__":
    # === Config ===
    csv_path = "../data/final_with_all_data.csv"
    sumo_config = "osm.sumocfg"
    log_dir = "./vlog"
    checkpoint_dir = "./vcheckpoints/3"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Load data ===
    route_data = load_routes(csv_path)

    # ✅ Vectorized env setup
    NUM_ENVS = 8  # Adjust based on your CPU
    env = SubprocVecEnv([
        make_env(i, sumo_config, route_data, csv_path)
        for i in range(NUM_ENVS)
    ])

    # === Noise ===
    n_actions = env.get_attr("action_space")[0].shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.0 * np.ones(n_actions))

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path=checkpoint_dir,
        name_prefix="vsac"
    )
    tensorboard_callback = TensorboardRewardLogger()

    # === Model ===
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        batch_size=512,
        learning_starts=100,
        gradient_steps=-1,
        learning_rate=1e-3,
        train_freq=(1, "step"),
        buffer_size=1_000_000,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=log_dir
    )

    # === Train ===
    model.learn(
        total_timesteps=50000,
        callback=[checkpoint_callback, tensorboard_callback]
    )

    # === Save ===
    model.save("vsac")
    print("✅ Model saved as sac_multi_route.zip")
