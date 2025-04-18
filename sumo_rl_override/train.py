from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import traci
from pathlib import Path
from sumo_rl.environment.env import SumoEnvironment

class GymnasiumWrapper(gym.Env):
    def __init__(self, **kwargs):
        # Start the environment
        self.env = SumoEnvironment(**kwargs)

        self.ts_id = self.env.ts_ids[6]  # or change to 6 if you're sure it's valid

        # Start simulation briefly to fetch observation shape and phase info
        temp_obs = self.env.reset()

        if isinstance(temp_obs, dict):
            obs_shape = temp_obs[self.ts_id].shape
        else:
            obs_shape = temp_obs.shape

        # ✅ Get number of available phases for this traffic light
        program_logics = traci.trafficlight.getAllProgramLogics(self.ts_id)
        phase_count = len(program_logics[0].phases)
        print(f"[Init] Traffic light '{self.ts_id}' has {phase_count} phases")

        # Setup spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(phase_count)

        # Close test sim
        self.env.close()

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return (obs[self.ts_id] if isinstance(obs, dict) else obs), {}

    def step(self, action):
        # Manually override the traffic light phase
        traci.trafficlight.setPhase(self.ts_id, int(action))

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


# === Instantiate the Environment ===

env = GymnasiumWrapper(
    net_file=str(Path("osm.net.xml").resolve()),
    route_file=str(Path("generated_flows.rou.xml").resolve()),
    use_gui=False,
    num_seconds=1500,
    yellow_time=3,
    min_green=5,
    max_green=90,
    fixed_ts=True,

)

# === Train PPO Agent ===
try:
    model = PPO.load("ppo_sumo_model_2.zip", env=env)
except:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard"
)
#
# model.n_epochs = 15
# model.ent_coef = 0.01
# model.vf_coef = 1.0
# model.gae_lambda = 0.98
# model.clip_range = 0.2
# model.learning_rate = 1e-4

model.learn(total_timesteps=50000,
            tb_log_name="run_2"
            )
model.save("ppo_sumo_model_3")
