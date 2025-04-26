from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

import gymnasium as gym
import numpy as np
import traci
from pathlib import Path
from sumo_rl.environment.env import SumoEnvironment
from gymnasium.spaces import Box


class MultiTLEnvSingleAgent(gym.Env):
    def __init__(self, model=None, **kwargs):
        self.env = SumoEnvironment(**kwargs)
        self.model = model
        self.ts_ids = self.env.ts_ids
        temp_obs = self.env.reset()
        obs_concat = np.concatenate([temp_obs[ts].flatten() for ts in self.ts_ids])

        self.min_duration = 5
        self.max_duration = 90
        self.phase_indices_per_tl = {}
        self.default_durations = []
        total_action_count = 0

        for ts_id in self.ts_ids:
            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            phases = logic.phases
            adjustable_indices = [i for i, phase in enumerate(phases) if 'y' not in phase.state.lower()]
            self.phase_indices_per_tl[ts_id] = adjustable_indices
            for idx in adjustable_indices:
                self.default_durations.append(phases[idx].duration)
            total_action_count += len(adjustable_indices)

        self.total_action_count = total_action_count
        self.action_space = Box(low=-1.0, high=1.0, shape=(total_action_count,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_concat.shape[0] + total_action_count,),
            dtype=np.float32
        )
        print(f"[Init] TLs: {len(self.ts_ids)}, adjustable phases per TL: {self.phase_indices_per_tl}")
        self.env.close()

    def _get_augmented_observation(self, obs):
        tl_durations = []

        for ts_id in self.ts_ids:
            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            durations = [phase.duration for i, phase in enumerate(logic.phases)
                         if i in self.phase_indices_per_tl[ts_id]]
            tl_durations.extend(durations)

        base_obs = np.concatenate([obs[ts].flatten() for ts in self.ts_ids])
        aug_obs = np.concatenate([base_obs, np.array(tl_durations, dtype=np.float32)])

        return aug_obs

    def set_model(self, model):
        self.model = model

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        obs_concat = self._get_augmented_observation(obs)

        if self.model is not None:
            try:
                action, _ = self.model.predict(
                    obs_concat,
                    deterministic=True,
                    state=None,
                    episode_start=np.array([True])
                )

                scale = 0.2
                scaled_actions = []
                for i, a in enumerate(action):
                    base = self.default_durations[i]
                    delta = scale * base
                    adj = base + a * delta
                    clipped = float(np.clip(adj, self.min_duration, self.max_duration))
                    scaled_actions.append(clipped)

                idx = 0
                for ts_id in self.ts_ids:
                    phase_indices = self.phase_indices_per_tl[ts_id]
                    count = len(phase_indices)
                    durations = scaled_actions[idx: idx + count]
                    idx += count

                    logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
                    phases = logic.phases
                    for phase_idx, duration in zip(phase_indices, durations):
                        phases[phase_idx].duration = duration

                    logic.phases = phases
                    traci.trafficlight.setProgramLogic(ts_id, logic)
            except Exception as e:
                print(f"[Error in model.predict]: {e}")

        return obs_concat, {}

    def step(self, action):
        obs, rewards, dones, infos = self.env.step({ts_id: 0 for ts_id in self.ts_ids})

        num_vehicles = traci.simulation.getMinExpectedNumber()
        sim_done = num_vehicles == 0 or self.env.sim_step >= self.env.sim_max_time
        done = sim_done or all(dones.values())
        obs_concat = self._get_augmented_observation(obs)

        try:
            total_wait_time = sum(
                traci.edge.getWaitingTime(edge_id)
                for edge_id in traci.edge.getIDList()
            )
            reward_sum = -total_wait_time / 1000
        except Exception as e:
            print(f"[Reward Error] {e}")
            reward_sum = 0

        if self.env.sim_step % 1000 == 0:
            print(f"[Step {self.env.sim_step}] Total Wait Time: {total_wait_time:.2f} → Reward: {reward_sum:.4f}")

        return obs_concat, reward_sum, done, False, {}

    def render(self):
        pass


if __name__ == "__main__":
    base_env = MultiTLEnvSingleAgent(
        net_file=str(Path("osm.net.xml").resolve()),
        route_file=str(Path("routes.rou.xml").resolve()),
        use_gui=False,
        num_seconds=2000,
        yellow_time=3,
        min_green=5,
        max_green=90,
        fixed_ts=True,
    )

    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])

    try:
        model = RecurrentPPO.load("model_lstm__1.zip", env=env)
    except:
        print("Training new LSTM-based model...")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_tensorboard",
            learning_rate=5e-5,
            gamma=0.98,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.05,
            n_steps=64,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=1
            )
        )

    base_env.set_model(model)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints_lstm/4/",
        name_prefix="ppo_tl_model_lstm"
    )

    try:
        model.learn(total_timesteps=100_000, tb_log_name="multi_tl_lstm_run", callback=checkpoint_callback)
    except Exception as e:
        print(e)
    finally:
        model.save("model_lstm_2_1")
