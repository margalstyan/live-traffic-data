import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from model import MultiRouteSUMOGymEnv


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


class TensorboardRewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_idx = 0
        self.rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            reward = self.locals["rewards"][0]
            self.rewards.append(reward)
            self.logger.record("rollout/episode_reward", reward)
            self.logger.record("rollout/ep_len_mean", 1.0)  # Always one step per episode
            self.logger.record("rollout/ep_rew_mean", np.mean(self.rewards[-100:]))
            self.episode_idx += 1
        return True


if __name__ == "__main__":
    # === Config ===
    csv_path = "../data/final_with_all_data.csv"
    sumo_config = "osm.sumocfg"
    log_dir = "./sac1_multi_logs"
    checkpoint_dir = "./sac_checkpoints/3"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Load data ===
    route_data = load_routes(csv_path)
    env = MultiRouteSUMOGymEnv(
        sumo_config_path=sumo_config,
        route_data_list=route_data,
        route_file_path="routes.rou.xml",
        tripinfo_path="tripinfo.xml",
        sumo_gui=False,
        csv_path=csv_path  # used for random timestamp selection
    )

    # === Noise ===
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.0 * np.ones(n_actions))

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path=checkpoint_dir,
        name_prefix="sac1_model"
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
        learning_rate=3e-4,
        train_freq=(1, "episode"),
        buffer_size=500_000,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=log_dir
    )
    # === Train ===
    model.learn(
        total_timesteps=50000,
        callback=[checkpoint_callback, tensorboard_callback]
    )

    # === Save ===
    model.save("sac1_multi_route")
    print("✅ Model saved as sac_multi_route.zip")
