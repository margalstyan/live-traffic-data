from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import traci
from lxml import etree

from simulation.generate_rou_single import generate_routes_for_next_timestamp


from gymnasium import Env, spaces
import numpy as np
import traci
from lxml import etree
import torch
from simulation.generate_rou_single import generate_routes_for_next_timestamp


class SUMOGymEnv(Env):
    def __init__(self, sumo_config_path, net_file_path, tls_id, use_gui=False, max_steps=300, tensorboard_writer=None):
        super(SUMOGymEnv, self).__init__()
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_config_path = sumo_config_path
        self.net_file_path = net_file_path
        self.tls_id = tls_id
        self.max_steps = max_steps
        self.current_step = 0
        self.writer = tensorboard_writer
        self.policy = None
        self.total_reward = 0
        self.last_durations = None

        self._parse_phases()
        obs = self._get_observation()
        self.state_dim = obs.shape[0]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.trainable_phase_indices),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        generate_routes_for_next_timestamp()
        if traci.isLoaded():
            traci.close()

        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config_path]
        traci.start(sumo_cmd)
        self.current_step = 0
        self.total_reward = 0

        traci.simulationStep()  # warm-up to populate obs
        obs = self._get_observation()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        if self.policy is not None:
            raise RuntimeError("In training mode, env should not use self.policy to predict actions.")

        # === Apply durations ===
        scaled_durations = 5 + action * (90 - 5)
        self._apply_action_durations(scaled_durations)
        self.last_durations = scaled_durations.tolist()

        # === Advance simulation ===
        traci.simulationStep()
        self.current_step += 1

        # === Observation and reward ===
        obs = self._get_observation()
        reward = self._calculate_reward()

        # === Episode termination logic ===
        terminated = self.current_step >= self.max_steps

        # Example: truncate if no vehicles left
        active_vehicles = traci.vehicle.getIDCount()
        truncated = (active_vehicles == 0) and not terminated

        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, {}

    def _parse_phases(self):
        tree = etree.parse(self.net_file_path)
        logic = tree.xpath(f"//tlLogic[@id='{self.tls_id}']")[0]

        self.all_phases = []
        self.yellow_phase_indices = []
        self.trainable_phase_indices = []

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

        traci.simulationStep()
        obs = self._get_observation()
        self.state_dim = obs.shape[0]

        traci.close()

    def _get_observation(self):
        if not traci.isLoaded():
            traci.start([self.sumo_binary, "-c", self.sumo_config_path])
        lane_ids = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.tls_id)))

        # Raw features
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids]
        waiting_times = [traci.lane.getWaitingTime(lane) for lane in lane_ids]

        # === Manual Normalization ===
        max_queue = 20  # e.g., 20 cars is high queue length
        max_wait = 300  # e.g., 5 minutes is high waiting time

        queue_lengths = [min(q, max_queue) / max_queue for q in queue_lengths]
        waiting_times = [min(w, max_wait) / max_wait for w in waiting_times]

        # Phase encoding
        num_phases = len(self.all_phases)
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        phase_onehot = np.zeros(num_phases)
        phase_onehot[current_phase] = 1

        # Final obs
        obs = np.array(queue_lengths + waiting_times + list(phase_onehot), dtype=np.float32)
        return obs

    def _apply_action_durations(self, durations):
        assert len(durations) == len(self.trainable_phase_indices)
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        new_phases = []

        dur_idx = 0
        for i, phase in enumerate(logic.phases):
            if i in self.trainable_phase_indices:
                duration = int(np.clip(durations[dur_idx], 5, 90))
                dur_idx += 1
            else:
                duration = 3  # fixed yellow phase

            new_phase = traci.trafficlight.Phase(duration, phase.state)
            new_phases.append(new_phase)

        new_logic = traci.trafficlight.Logic(
            programID="custom",  # required to switch
            type=logic.type,
            currentPhaseIndex=logic.currentPhaseIndex,
            phases=new_phases
        )

        traci.trafficlight.setProgramLogic(self.tls_id, new_logic)
        traci.trafficlight.setProgram(self.tls_id, "custom")
        traci.simulationStep()

    def _calculate_reward(self):
        lane_ids = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.tls_id)))

        # === Total waiting time ===
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lane_ids)
        max_wait_per_lane = 300  # assume 5 min is max wait worth penalizing
        max_total_wait = max_wait_per_lane * len(lane_ids)
        normalized_wait = min(total_waiting_time, max_total_wait) / max_total_wait  # [0, 1]

        # === Average queue length ===
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids]
        avg_queue_length = np.mean(queue_lengths)
        max_queue_length = 20  # reasonable upper bound
        normalized_queue = min(avg_queue_length, max_queue_length) / max_queue_length  # [0, 1]

        # === Reward shaping: weighted sum ===
        wait_weight = 0.7
        queue_weight = 0.3

        reward = - (wait_weight * normalized_wait + queue_weight * normalized_queue)  # [-1, 0]

        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_current_green_durations(self):
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        return [phase.duration for i, phase in enumerate(logic.phases) if i in self.trainable_phase_indices]
