from gymnasium import Env
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
from lxml import etree

from simulation.generate_rou_single import generate_random_routes



class SUMOGymEnv(Env):
    def __init__(self, sumo_config_path, net_file_path, tls_id, use_gui=False, max_steps=300,
                 tensorboard_writer=None, route_file_path="../sumo_rl_single/routes.rou.xml"):
        super(SUMOGymEnv, self).__init__()
        self.last_episode_steps = None
        self.route_file_path = route_file_path
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
        self.observation_space = spaces.Box(low=0, high=1, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.trainable_phase_indices),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if traci.isLoaded():
            traci.close()
        self.current_step = 0
        self.total_reward = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action, *args, **kwargs):
        # === Generate new routes
        generate_random_routes(output_file=self.route_file_path, junction_id=self.tls_id)

        # === Restart SUMO with tripinfo.xml output
        if traci.isLoaded():
            traci.close()

        trip_info_out = None
        if "tripinfo" in kwargs:
            trip_info_out = kwargs["tripinfo"]
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config_path,
            "--route-files", self.route_file_path,
            "--tripinfo-output", trip_info_out or f"xml/tripinfo_{os.getpid()}.xml",
        ]

        traci.start(sumo_cmd)

        # === Apply static phase durations once
        scaled_durations = 10 + action * (60 - 10)
        self._apply_action_durations(scaled_durations)
        self.last_durations = scaled_durations.tolist()

        # === Run full simulation episode
        total_reward = 0
        actual_steps = 0
        for step in range(self.max_steps):
            traci.simulationStep()
            reward = self._calculate_reward()
            total_reward += reward
            actual_steps += 1

            if traci.simulation.getMinExpectedNumber() == 0:
                break  # ✅ No more vehicles left

        traci.close()
        self.last_episode_steps = actual_steps

        obs = self._get_observation()
        terminated = True
        truncated = (actual_steps < self.max_steps)
        avg_reward = total_reward / actual_steps
        info = {
            "ep_length_real": actual_steps,
            "green_durations": self.last_durations,
            "ep_rew_mean": avg_reward,
        }
        return obs, float(avg_reward), terminated, truncated, info

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
        traci.close()

    def _get_observation(self):
        if not traci.isLoaded():
            traci.start([self.sumo_binary, "-c", self.sumo_config_path])

        lane_ids = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.tls_id)))
        lane_vehicle_counts = {lane: 0 for lane in lane_ids}

        # Parse expected vehicle flows from the route file
        if os.path.exists(self.route_file_path):
            from lxml import etree
            tree = etree.parse(self.route_file_path)
            route_elements = tree.xpath("//route")
            flow_elements = tree.xpath("//flow")

            # Build route_id → list of edges mapping
            route_id_to_edges = {
                route.attrib["id"]: route.attrib["edges"].split()
                for route in route_elements if "id" in route.attrib and "edges" in route.attrib
            }

            for flow in flow_elements:
                route_id = flow.attrib.get("route", "")
                number = int(flow.attrib.get("number", 0))
                edges = route_id_to_edges.get(route_id, [])

                for edge in edges:
                    try:
                        num_lanes = traci.edge.getLaneNumber(edge)
                        for i in range(num_lanes):
                            lane_id = f"{edge}_{i}"
                            if lane_id in lane_vehicle_counts:
                                lane_vehicle_counts[lane_id] += number
                    except Exception:
                        continue  # Skip invalid edge

        max_expected = 50
        norm_expected_counts = [min(lane_vehicle_counts[lane], max_expected) / max_expected for lane in lane_ids]

        # Phase encoding (already one-hot)
        num_phases = len(self.all_phases)
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        phase_onehot = np.zeros(num_phases)
        phase_onehot[current_phase] = 1

        # Green durations normalization
        green_durations = self.get_current_green_durations()
        norm_durations = [dur / 60 for dur in green_durations]

        # Final normalized observation
        obs = np.array(norm_expected_counts + list(phase_onehot) + norm_durations, dtype=np.float32)
        return obs

    def _apply_action_durations(self, durations):
        assert len(durations) == len(self.trainable_phase_indices)
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        new_phases = []

        dur_idx = 0
        for i, phase in enumerate(logic.phases):
            if i in self.trainable_phase_indices:
                duration = int(np.clip(durations[dur_idx], 10, 60))
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
        wait_weight = 0.5
        queue_weight = 0.5

        reward = - (wait_weight * normalized_wait + queue_weight * normalized_queue)  # [-1, 0]
        reward = np.clip(reward, -1, 0)

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


