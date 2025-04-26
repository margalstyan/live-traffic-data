from stable_baselines3 import DDPG
import gymnasium as gym
import numpy as np
import traci
from pathlib import Path
from sumo_rl.environment.env import SumoEnvironment
from gymnasium.spaces import Box
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


class MultiTLEnvSingleAgent(gym.Env):
    def __init__(self, model=None, **kwargs):
        self.env = SumoEnvironment(**kwargs)
        self.model = model  # Store the PPO model explicitly
        self.ts_ids = self.env.ts_ids
        temp_obs = self.env.reset()

        obs_concat = np.concatenate([temp_obs[ts].flatten() for ts in self.ts_ids])
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_concat.shape,
            dtype=np.float32
        )

        self.min_duration = 5
        self.max_duration = 90

        self.phase_indices_per_tl = {}
        total_action_count = 0

        for ts_id in self.ts_ids:
            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            phases = logic.phases
            adjustable_indices = [i for i, phase in enumerate(phases) if 'y' not in phase.state.lower()]
            self.phase_indices_per_tl[ts_id] = adjustable_indices
            total_action_count += len(adjustable_indices)

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_count,),
            dtype=np.float32
        )

        print(f"[Init] TLs: {len(self.ts_ids)}, adjustable phases per TL: {self.phase_indices_per_tl}")

        self.env.close()

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()

        # Get concatenated observation
        obs_concat = np.concatenate([obs[ts].flatten() for ts in self.ts_ids])

        # Predict action ONCE per episode (initial action for durations)
        action, _ = self.model.predict(obs_concat, deterministic=True)

        # Scale actions explicitly from [-1, 1] to [min_duration, max_duration]
        scaled_actions = self.min_duration + (action + 1.0) * (self.max_duration - self.min_duration) / 2.0

        # Set durations per TL once at the start
        idx = 0
        for ts_id in self.ts_ids:
            phase_indices = self.phase_indices_per_tl[ts_id]
            count = len(phase_indices)
            durations = scaled_actions[idx: idx + count]
            idx += count

            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            phases = logic.phases

            for phase_idx, duration in zip(phase_indices, durations):
                clipped_duration = float(np.clip(duration, self.min_duration, self.max_duration))
                phases[phase_idx].duration = clipped_duration

            logic.phases = phases
            traci.trafficlight.setProgramLogic(ts_id, logic)

        return obs_concat, {}

    def step(self, action):
        obs, rewards, dones, infos = self.env.step({ts_id: 0 for ts_id in self.ts_ids})

        num_vehicles = traci.simulation.getMinExpectedNumber()
        sim_done = num_vehicles == 0 or self.env.sim_step >= self.env.sim_max_time
        done = sim_done or all(dones.values())

        obs_concat = np.concatenate([obs[ts].flatten() for ts in self.ts_ids])
        reward_sum = sum(rewards.values())
        if reward_sum == 0:
            reward_sum = 1/155
        return obs_concat, reward_sum, done, False, {}

    def render(self):
        pass


if __name__ == "__main__":
    env = MultiTLEnvSingleAgent(
        net_file=str(Path("osm.net.xml").resolve()),
        route_file=str(Path("routes.rou.xml").resolve()),
        use_gui=False,
        num_seconds=2000,
        yellow_time=3,
        min_green=5,
        max_green=90,
        fixed_ts=True,
    )
    env = Monitor(env)

    try:
        model = DDPG.load("model_ddpg_1.zip",
                         env=env,
                         learning_rate=3e-4,
                         clip_range=0.1,
                         ent_coef=0.001,
                         n_epochs=20)

    except:
        print("Training new model...")
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_tensorboard",
            learning_rate=3e-4,
        )

    # Pass model into env
    env.env.model = model

    # ✅ Checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/ddpg/",
        name_prefix="ppo_tl_model"
    )

    try:
        model.learn(total_timesteps=100_000, tb_log_name="multi_tl_run", callback=checkpoint_callback)
    except Exception as e:
        print(e)
    finally:
        model.save("model_ddpg_1")
