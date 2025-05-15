from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from sumo_rl.environment.traffic_signal import TrafficSignal

from sumo_rl.environment.env import SumoEnvironment
from sumolib.net import Phase

class GymnasiumWrapper(gym.Env):
    def __init__(self, **kwargs):
        self.env = SumoEnvironment(**kwargs)
        self.ts_id = self.env.ts_ids[6]


        obs = self.env.reset()
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(obs[self.ts_id]),),
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return obs[self.ts_id], {}

    def step(self, action):
        # map agent's action (0-3) to actual green phase index
        obs, reward, done, info = self.env.step({self.ts_id: action})
        if self.env.sim_step >= 1000:
            done[self.ts_id] = True

        return (
            obs[self.ts_id],
            reward[self.ts_id],
            done[self.ts_id],
            False,
            info.get(self.ts_id, {})
        )

    def render(self):
        pass

    def close(self):
        self.env.close()


# Instantiate and train
env = GymnasiumWrapper(
    net_file="osm.net.xml",
    route_file="routes.rou.xml",
    use_gui=True,
    num_seconds=1000,
    yellow_time=3,
    min_green=5,
    max_green=90,
    fixed_ts=True
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_sumo_model")
