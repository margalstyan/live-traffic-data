import os
import numpy as np
import pandas as pd
import shutil
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from vmodel import MultiRouteSUMOGymEnv


# === Load routes from CSV
def load_routes(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    return [{
        "id": i,
        "edges": [str(row["from_edge"]), str(row["to_edge"])],
        "duration_without_traffic": float(row["duration_without_traffic"]),
        "distance": float(row["distance"])
    } for i, row in df.iterrows()]


# === Environment factory
def make_env(instance_id, sumo_config, route_data, csv_path):
    def _init():
        return MultiRouteSUMOGymEnv(
            sumo_config_path=sumo_config,
            route_data_list=route_data,
            route_file_path=f"xml/routes_{instance_id}.rou.xml",
            tripinfo_path=f"xml/tripinfo_{instance_id}.xml",
            sumo_gui=False,
            csv_path=csv_path,
            instance_id=instance_id
        )

    return _init


# === TensorBoard reward logger
class TensorboardRewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if dones is not None and rewards is not None:
            for done, reward in zip(dones, rewards):
                if done:
                    self.rewards.append(reward)
                    self.logger.record("rollout/episode_reward", reward)
                    self.logger.record("rollout/ep_len_mean", 1.0)
                    self.logger.record("rollout/ep_rew_mean", np.mean(self.rewards[-100:]))
        self.logger.record("train/current_lr", self.model.learning_rate)
        return True


# === Custom safe checkpoint callback
class CheckpointWithReplayBufferCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="vsac", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_str = f"{self.num_timesteps}_steps"
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{step_str}.zip")
            buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{step_str}_buffer.pkl")
            tmp_buffer_path = buffer_path + ".tmp"

            self.model.save(model_path)

            # try:
            # self.model.save_replay_buffer(tmp_buffer_path)
            # shutil.move(tmp_buffer_path, buffer_path)
            # if self.verbose:
            #     print(f"📌 Model saved to {model_path}")
            #     print(f"💾 Replay buffer safely saved to {buffer_path}")
            # except Exception as e:
            #     print(f"❌ Failed to save replay buffer: {e}")
            #     if os.path.exists(tmp_buffer_path):
            #         os.remove(tmp_buffer_path)

        return True

def linear_lr_schedule(progress):
    """
    progress: float in [1.0, 0.0] — 1.0 is beginning, 0.0 is end of training
    """
    if progress > 0.4:
        return 1e-3
    elif progress > 0.2:
        return 3e-4
    else:
        return 1e-4

if __name__ == "__main__":
    # === Config
    csv_path = "../data/final_with_all_data.csv"
    sumo_config = "osm.sumocfg"
    log_dir = "./vlog"
    checkpoint_dir = "./vcheckpoints/10"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Load route data
    route_data = load_routes(csv_path)

    # === Vectorized environment
    NUM_ENVS = 4
    env = SubprocVecEnv([
        make_env(i, sumo_config, route_data, csv_path)
        for i in range(NUM_ENVS)
    ])

    # === Action noise
    n_actions = env.get_attr("action_space")[0].shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.0 * np.ones(n_actions))

    # === Create model with fine-tune parameters
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        batch_size=512,
        learning_starts=512,
        gradient_steps=-1,
        learning_rate=linear_lr_schedule,
        train_freq=(1, "step"),
        buffer_size=500_000,
        policy_kwargs={"net_arch": [512, 512, 256]},
        ent_coef="auto_0.5",
        tau=0.005,
        tensorboard_log=log_dir
    )

    # === Load model weights only (no buffer)
    # resume_step = 4000
    # model_path = os.path.join("./vcheckpoints/9", f"vsac_{resume_step}_steps.zip")
    # model.load_replay_buffer(buffer_path)
    # if os.path.exists(model_path):
    #     print(f"🔁 Loading model weights from {model_path}")
    #     model.set_parameters(model_path)

    # ✅ Sync target critic for stability
    # model.policy.critic_target.load_state_dict(model.policy.critic.state_dict())
    # else:
    #     print("⚠️ Checkpoint not found. Training from scratch.")

    # === Callbacks
    checkpoint_callback = CheckpointWithReplayBufferCallback(
        save_freq=25,
        save_path=checkpoint_dir,
        name_prefix="vsac",
        verbose=1
    )
    tensorboard_callback = TensorboardRewardLogger()

    # === Train
    model.learn(
        total_timesteps=10000,
        callback=[checkpoint_callback, tensorboard_callback]
    )

    # === Final save
    model.save("vsac_finetuned")
    model.save_replay_buffer("vsac_finetuned_buffer.pkl")
    print("✅ Model and buffer saved as vsac_finetuned.zip / .pkl")
