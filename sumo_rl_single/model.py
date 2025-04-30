from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import traci
from lxml import etree

from simulation.generate_rou_single import generate_routes_for_next_timestamp


class SUMOGymEnv(gym.Env):
    def __init__(self, sumo_config_path, net_file_path, tls_id, use_gui=False, max_steps=300, tensorboard_writer=None):
        super(SUMOGymEnv, self).__init__()
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_config_path = sumo_config_path
        self.net_file_path = net_file_path
        self.tls_id = tls_id
        self.max_steps = max_steps
        self.current_step = 0
        self._parse_phases()
        self.writer = tensorboard_writer
        self.policy = None
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.trainable_phase_indices),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        generate_routes_for_next_timestamp()
        if traci.isLoaded():
            traci.close()

        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config_path]
        traci.start(sumo_cmd)
        self.current_step = 0
        obs = self._get_observation()

        # === Predict new TL durations using current policy (once per episode)
        if self.policy is not None:
            import torch
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.policy.device)
            action, _ = self.policy.predict(obs_tensor, deterministic=False)
            scaled_durations = 5 + action[0] * (90 - 5)  # scale from [0,1] to [5,90]
            self._apply_action_durations(scaled_durations)

        if self.writer:
            durations = self.get_current_green_durations()
            for i, dur in enumerate(durations):
                self.writer.add_scalar(f"phases/green_phase_{i}", dur, self.current_step)

        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps
        terminated = self.current_step >= self.max_steps
        truncated = False  # Optional logic, can also be: time limit or manual cutoff
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, {}


    def _parse_phases(self):
        """Parse the net.xml and extract phase info"""
        tree = etree.parse(self.net_file_path)
        logic = tree.xpath(f"//tlLogic[@id='{self.tls_id}']")[0]

        self.all_phases = []  # list of (index, phase_str)
        self.yellow_phase_indices = []  # e.g., [1, 3, 5]
        self.trainable_phase_indices = []  # e.g., [0, 2, 4]

        for i, phase in enumerate(logic.findall("phase")):
            state = phase.attrib["state"]
            self.all_phases.append((i, state))
            if 'y' in state:
                self.yellow_phase_indices.append(i)
            else:
                self.trainable_phase_indices.append(i)

        self.action_dim = len(self.trainable_phase_indices)

        if traci.isLoaded():
            traci.close()

        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config_path]
        traci.start(sumo_cmd)

        obs = self._get_observation()
        self.state_dim = obs.shape[0]

        traci.close()

    def _get_observation(self):
        lane_ids = traci.trafficlight.getControlledLanes(self.tls_id)
        lane_ids = list(dict.fromkeys(lane_ids))  # remove duplicates while preserving order

        # Feature 1: queue length
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids]

        # Feature 2: total waiting time
        waiting_times = [traci.lane.getWaitingTime(lane) for lane in lane_ids]

        # Feature 3: current phase (one-hot)
        num_phases = len(self.all_phases)
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        phase_onehot = np.zeros(num_phases)
        phase_onehot[current_phase] = 1

        obs = np.array(queue_lengths + waiting_times + list(phase_onehot), dtype=np.float32)
        return obs

    def _apply_action_durations(self, durations):
        """
        Apply model-predicted durations to green/red phases,
        and fix yellow phases to 3s.
        `durations` is a list/array of len = len(self.trainable_phase_indices)
        """

        assert len(durations) == len(self.trainable_phase_indices)

        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]  # get current program
        new_phases = []

        dur_idx = 0
        for i, phase in enumerate(logic.phases):
            if i in self.trainable_phase_indices:
                duration = int(np.clip(durations[dur_idx], 5, 90))
                dur_idx += 1
            else:
                duration = 3  # yellow phases fixed

            new_phase = traci.trafficlight.Phase(duration, phase.state)
            new_phases.append(new_phase)

        new_logic = traci.trafficlight.Logic(
            logic.programID,
            logic.type,
            logic.currentPhaseIndex,
            new_phases
        )

        traci.trafficlight.setProgramLogic(self.tls_id, new_logic)

    def _calculate_reward(self):
        """
        Calculate reward as negative total waiting time across all controlled lanes.
        """
        lane_ids = traci.trafficlight.getControlledLanes(self.tls_id)
        lane_ids = list(dict.fromkeys(lane_ids))  # remove duplicates

        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lane_ids)
        reward = -total_waiting_time  # we want to minimize it
        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()
        else:
            print("⚠️ Warning: SUMO is not loaded. Cannot close.")

    def get_current_green_durations(self):
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        durations = [phase.duration for i, phase in enumerate(logic.phases) if i in self.trainable_phase_indices]
        return durations