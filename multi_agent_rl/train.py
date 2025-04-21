from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import traci
from pathlib import Path
from sumo_rl.environment.env import SumoEnvironment
from gymnasium.spaces import Box
from stable_baselines3.common.monitor import Monitor

class MultiTLEnvSingleAgent(gym.Env):
    def __init__(self, **kwargs):
        self.env = SumoEnvironment(**kwargs)
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

        # Clearly identify GREEN and RED phases (not YELLOW)
        self.phase_indices_per_tl = {}
        total_action_count = 0

        for ts_id in self.ts_ids:
            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            phases = logic.phases
            adjustable_indices = [i for i, phase in enumerate(phases) if 'y' not in phase.state.lower()]  # Skip yellow phases
            self.phase_indices_per_tl[ts_id] = adjustable_indices
            total_action_count += len(adjustable_indices)

        # Set PPO action space explicitly to normalized range [-1, 1]
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_count,),
            dtype=np.float32
        )

        print(f"[Init] TLs: {len(self.ts_ids)}, adjustable phases per TL: {self.phase_indices_per_tl}")

        self.env.close()

    def reset(self, *, seed=None, options=None):
        self.env.close()
        obs = self.env.reset()
        obs_concat = np.concatenate([obs[ts].flatten() for ts in self.ts_ids])
        return obs_concat, {}

    def step(self, action):
        # Scale PPO outputs [-1,1] explicitly to [min_duration, max_duration]
        scaled_actions = self.min_duration + (action + 1.0) * (self.max_duration - self.min_duration) / 2.0

        idx = 0
        durations_per_tl = {}
        for ts_id in self.ts_ids:
            phase_indices = self.phase_indices_per_tl[ts_id]
            count = len(phase_indices)
            durations_per_tl[ts_id] = scaled_actions[idx: idx + count]
            idx += count

        for ts_id, durations in durations_per_tl.items():
            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            phases = logic.phases

            # Update only green/red phases explicitly
            for phase_idx, duration in zip(self.phase_indices_per_tl[ts_id], durations):
                clipped_duration = float(np.clip(duration, self.min_duration, self.max_duration))
                phases[phase_idx].duration = clipped_duration

            logic.phases = phases
            traci.trafficlight.setProgramLogic(ts_id, logic)

        obs, rewards, dones, infos = self.env.step({ts_id: 0 for ts_id in self.ts_ids})

        num_vehicles = traci.simulation.getMinExpectedNumber()
        sim_done = num_vehicles == 0 or self.env.sim_step >= self.env.sim_max_time
        done = sim_done or all(dones.values())

        if done:
            print(f"🚦 Episode ended at step {self.env.sim_step}.")
            self.env.close()

        obs_concat = np.concatenate([obs[ts].flatten() for ts in self.ts_ids])
        reward_sum = sum(rewards.values())

        return obs_concat, reward_sum, done, False, {}

    def render(self):
        pass

    def close(self):
        self.env.close()

env = MultiTLEnvSingleAgent(
    net_file=str(Path("osm.net.xml").resolve()),
    route_file=str(Path("routes.rou.xml").resolve()),
    use_gui=False,
    num_seconds=1500,
    yellow_time=3,
    min_green=5,
    max_green=90,
    fixed_ts=True,
)
env = Monitor(env)

try:
    model = PPO.load("model_6.zip", env=env)
except:
    print("Training new model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard",
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64,
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.98,
        vf_coef=0.5,
        normalize_advantage=True
    )

try:
    model.learn(total_timesteps=40960, tb_log_name="multi_tl_run")
except Exception as e:
    print(e)
finally:
    model.save("model_7")
