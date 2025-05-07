import os
import time

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from lxml import etree
from typing import Dict, List
from simulation.generate_rou_single import generate_random_routes

class MultiSUMOGymEnv(gym.Env):
    def __init__(self, sumo_config_path, net_file_path, tls_ids: List[str],
                 use_gui=False, max_steps=300, route_file_path="../sumo_rl_single/routes.rou.xml", tripinfo_path=None):
        super().__init__()
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_config_path = sumo_config_path
        self.net_file_path = net_file_path
        self.tls_ids = tls_ids
        self.route_file_path = route_file_path
        self.max_steps = max_steps
        self.tripinfo_path = tripinfo_path or f"xml/tripinfo_{os.getpid()}.xml"
        self.trainable_phase_indices = {}
        for tls_id in self.tls_ids:
            self.trainable_phase_indices[tls_id] = self._parse_trainable_phases(tls_id)

        self.observation_space = spaces.Dict({
            tls_id: spaces.Box(low=0, high=1, shape=(3 + self._estimate_phase_count(tls_id) + len(self.trainable_phase_indices[tls_id]),), dtype=np.float32)
            for tls_id in self.tls_ids
        })

        self.action_space = spaces.Dict({
            tls_id: spaces.Box(low=0, high=1, shape=(len(self.trainable_phase_indices[tls_id]),), dtype=np.float32)
            for tls_id in self.tls_ids
        })

    def reset(self, *, seed=None, options=None):
        # === Generate new routes
        random_junction_id = np.random.choice(self.tls_ids)
        generate_random_routes(output_file=self.route_file_path, junction_id=random_junction_id)
        if traci.isLoaded():
            traci.close()

        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config_path, "--route-files", self.route_file_path]
        traci.start(sumo_cmd)

        for _ in range(10):
            traci.simulationStep()

        obs = {}
        for tls_id in self.tls_ids:
            obs[tls_id] = np.zeros(self.observation_space[tls_id].shape, dtype=np.float32)

        traci.close()
        return obs, {}

    def step(self, actions: Dict[str, np.ndarray]):
        if traci.isLoaded():
            traci.close()

        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config_path,
            "--route-files", self.route_file_path,
            "--tripinfo-output", self.tripinfo_path,
            "--no-step-log", "true",
        ]
        traci.start(sumo_cmd)

        for tls_id, action in actions.items():
            durations = 10 + action * (60 - 10)
            self._apply_action_durations(tls_id, durations)

        for _ in range(self.max_steps):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() == 0:
                break
        traci.close()
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config_path,
            "--no-step-log", "true",
        ]
        traci.start(sumo_cmd)
        tripinfo_stats = self._parse_tripinfo_per_tls(self.tripinfo_path)

        obs = {}
        rewards = {}
        dones = {}
        infos = {}
        if not traci.isLoaded():
            traci.start(sumo_cmd)

        for tls_id in self.tls_ids:
            stats = tripinfo_stats[tls_id]
            obs[tls_id] = self._get_episode_observation(tls_id, stats)
            rewards[tls_id] = self._calculate_reward(stats)
            dones[tls_id] = True
            infos[tls_id] = {}
        if traci.isLoaded():
            traci.close()
        return obs, rewards, dones, dones, infos

    def _parse_tripinfo_per_tls(self, tripinfo_path: str) -> Dict[str, Dict]:

        tls_stats = {
            tls_id: {"wait_sum": 0.0, "dur_sum": 0.0, "count": 0}
            for tls_id in self.tls_ids
        }
        time.sleep(1)
        while not os.path.exists(tripinfo_path):
            print(f"Tripinfo file {tripinfo_path} not found.")

        tls_lanes_map = {
            tls_id: set(traci.trafficlight.getControlledLanes(tls_id))
            for tls_id in self.tls_ids
        }
        if traci.isLoaded():
            traci.close()
        tree = etree.parse(tripinfo_path)
        root = tree.getroot()

        for trip in root.iter("tripinfo"):
            depart_lane = trip.attrib.get("departLane")
            arrival_lane = trip.attrib.get("arrivalLane")
            wait = float(trip.attrib["waitingTime"])
            dur = float(trip.attrib["duration"])

            for tls_id, lanes in tls_lanes_map.items():
                if (depart_lane in lanes) or (arrival_lane in lanes):
                    tls_stats[tls_id]["wait_sum"] += wait
                    tls_stats[tls_id]["dur_sum"] += dur
                    tls_stats[tls_id]["count"] += 1
                    break

        return tls_stats

    def _get_episode_observation(self, tls_id, stats):
        avg_wait = min(stats["wait_sum"] / stats["count"], 300) / 300 if stats["count"] else 0
        avg_duration = min(stats["dur_sum"] / stats["count"], 300) / 300 if stats["count"] else 0
        throughput = min(stats["count"], 500) / 500

        num_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_onehot = np.zeros(num_phases)
        phase_onehot[current_phase] = 1

        green_durations = self.get_current_green_durations(tls_id)
        norm_durations = [dur / 60 for dur in green_durations]

        return np.array([avg_wait, avg_duration, throughput] + list(phase_onehot) + norm_durations, dtype=np.float32)

    def _calculate_reward(self, stats):
        if stats["count"] == 0:
            return -1.0
        wait = stats["wait_sum"] / stats["count"]
        return float(np.clip(-wait / 300, -1, 0))

    def _parse_trainable_phases(self, tls_id):
        tree = etree.parse(self.net_file_path)
        logic = tree.xpath(f"//tlLogic[@id='{tls_id}']")[0]
        return [i for i, phase in enumerate(logic.findall("phase")) if 'y' not in phase.attrib["state"]]

    def _estimate_phase_count(self, tls_id):
        tree = etree.parse(self.net_file_path)
        return len(tree.xpath(f"//tlLogic[@id='{tls_id}']/phase"))

    def _apply_action_durations(self, tls_id, durations):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        new_phases = []
        dur_idx = 0
        for i, phase in enumerate(logic.phases):
            duration = 3 if 'y' in phase.state else int(np.clip(durations[dur_idx], 10, 60))
            if 'y' not in phase.state:
                dur_idx += 1
            new_phases.append(traci.trafficlight.Phase(duration, phase.state))
        new_logic = traci.trafficlight.Logic(
            programID="custom",
            type=logic.type,
            currentPhaseIndex=logic.currentPhaseIndex,
            phases=new_phases
        )
        traci.trafficlight.setProgramLogic(tls_id, new_logic)
        traci.trafficlight.setProgram(tls_id, "custom")

    def get_current_green_durations(self, tls_id):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        return [
            phase.duration
            for i, phase in enumerate(logic.phases)
            if i in self.trainable_phase_indices[tls_id]
        ]

    def close(self):
        if traci.isLoaded():
            traci.close()