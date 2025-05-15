import os
import time
import traci
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List
from lxml import etree
from tls_utils import get_tls_info, TrafficLightInfo
from simulation_rl.generate_routes import generate_routes_with_sac_model

EPISODE_DURATION = 3600
GREEN_PHASE_MIN = 10
GREEN_PHASE_MAX = 60
YELLOW_DURATION = 3
TRIPINFO_PATH = "tripinfo.xml"
ROUTE_OUTPUT_PATH = "routes_generated.rou.xml"
SAC_MODEL_PATH = "../simulation_rl/sac_checkpoints/1/sac1_model_1000_steps.zip"


class SUMOMultiAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sumo_cfg: str, route_csv: str, sumo_port: int = 8813):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.route_csv = route_csv
        self.tripinfo_path = TRIPINFO_PATH
        self.generated_route_path = ROUTE_OUTPUT_PATH
        self.tls_info_map: Dict[str, TrafficLightInfo] = get_tls_info()
        self.tls_ids = list(self.tls_info_map.keys())
        self.sumo_port = sumo_port

        self.agent_action_spaces = {
            tls_id: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.tls_info_map[tls_id].green_phases),),
                dtype=np.float32
            )
            for tls_id in self.tls_ids
        }

        self.agent_observation_spaces = {
            tls_id: spaces.Box(
                low=np.array([0, 0, 0, 0], dtype=np.float32),
                high=np.array([1, 1, 1, 1], dtype=np.float32),
                shape=(4,),
                dtype=np.float32
            )
            for tls_id in self.tls_ids
        }

    def reset(self, *, seed=None, options=None):
        traci.start(["sumo", "-c", self.sumo_cfg, "--no-step-log", "true"], port=self.sumo_port)
        self._generate_routes()

        if traci.isLoaded():
            traci.close()

        traci.start([
            "sumo", "-c", self.sumo_cfg,
            "--route-files", self.generated_route_path,
            "--no-step-log", "true",
            "--no-warnings", "true"
        ], port=self.sumo_port)

        obs = self._get_observations()
        return obs, {}

    def step(self, actions: Dict[str, np.ndarray]):
        if traci.isLoaded():
            traci.close()

        traci.start([
            "sumo", "-c", self.sumo_cfg,
            "--route-files", self.generated_route_path,
            "--tripinfo-output", self.tripinfo_path,
            "--no-step-log", "true",
            "--no-warnings", "true",
        ], port=self.sumo_port)

        for tls_id, action in actions.items():
            durations = GREEN_PHASE_MIN + action * (GREEN_PHASE_MAX - GREEN_PHASE_MIN)
            self._apply_action_durations(tls_id, durations)

        step_count = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step_count < EPISODE_DURATION:
            traci.simulationStep()
            step_count += 1

        traci.close()
        traci.start([
            "sumo", "-c", self.sumo_cfg,
            "--no-step-log", "true"
        ], port=self.sumo_port)
        rewards, infos = self._compute_rewards()
        obs = self._get_observations()

        done = {tls_id: True for tls_id in self.tls_ids}
        done["__all__"] = True

        if step_count >= EPISODE_DURATION:
            for tls_id in self.tls_ids:
                rewards[tls_id] += -0.1
                infos[tls_id][0]["timeout_penalty"] = -0.1
        if traci.isLoaded():
            traci.close()
        return obs, rewards, done, False, infos

    def _generate_routes(self):
        tls_edge_map = {
            tls_id: list(set(l.rsplit("_", 1)[0] for l in traci.trafficlight.getControlledLanes(tls_id)))
            for tls_id in self.tls_ids
        }
        result = generate_routes_with_sac_model(
            output_file=self.generated_route_path,
            model_path=SAC_MODEL_PATH,
            tls_edge_map=tls_edge_map
        )
        self.route_stats = result["tls_stats"]
        self.timestamp_column = result["timestamp_column"]  # e.g., 'duration_0830'

    def _get_observations(self) -> Dict[str, np.ndarray]:
        # Extract normalized timestamp from 'duration_0830'
        hhmm = self.timestamp_column.split('_')[-1]
        hour = int(hhmm[:2])
        minute = int(hhmm[2:])
        minutes_of_day = hour * 60 + minute
        normalized_time = minutes_of_day / 1440.0

        obs = {}
        for tls_id in self.tls_ids:
            stats = self.route_stats.get(tls_id, {
                "avg_distance": 500.0,
                "avg_duration": 150.0,
                "vehicle_count": 1
            })

            dist = min(stats["avg_distance"] / 1000.0, 1.0)
            dur = min(stats["avg_duration"] / 300.0, 1.0)
            count = min(stats["vehicle_count"], 3000) / 3000.0

            obs_vector = np.array([
                normalized_time,
                dist,
                dur,
                count
            ], dtype=np.float32)
            obs[tls_id] = obs_vector

        return obs

    def _apply_action_durations(self, tls_id: str, durations: np.ndarray):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        new_phases = []
        green_indices = self.tls_info_map[tls_id].green_phases
        dur_idx = 0

        for i, phase in enumerate(logic.phases):
            if i in green_indices:
                duration = int(np.clip(durations[dur_idx], GREEN_PHASE_MIN, GREEN_PHASE_MAX))
                new_phases.append(traci.trafficlight.Phase(duration, phase.state))
                dur_idx += 1
            elif "y" in phase.state:
                new_phases.append(traci.trafficlight.Phase(YELLOW_DURATION, phase.state))
            else:
                new_phases.append(phase)

        new_logic = traci.trafficlight.Logic(
            programID="custom",
            type=logic.type,
            currentPhaseIndex=logic.currentPhaseIndex,
            phases=new_phases
        )
        traci.trafficlight.setProgramLogic(tls_id, new_logic)
        traci.trafficlight.setProgram(tls_id, "custom")

    def _compute_rewards(self):
        time.sleep(1)
        while not os.path.exists(self.tripinfo_path):
            print(f"Waiting for tripinfo: {self.tripinfo_path}")
            time.sleep(0.1)

        tree = etree.parse(self.tripinfo_path)
        root = tree.getroot()

        tls_lanes_map = {
            tls_id: set(traci.trafficlight.getControlledLanes(tls_id))
            for tls_id in self.tls_ids
        }

        tls_stats = {tls_id: {"wait_sum": 0.0, "dur_sum": 0.0, "count": 0} for tls_id in self.tls_ids}

        for trip in root.iter("tripinfo"):
            wait = float(trip.attrib["waitingTime"])
            dur = float(trip.attrib["duration"])
            depart_lane = trip.attrib.get("departLane")
            arrival_lane = trip.attrib.get("arrivalLane")

            for tls_id, lanes in tls_lanes_map.items():
                if depart_lane in lanes or arrival_lane in lanes:
                    tls_stats[tls_id]["wait_sum"] += wait
                    tls_stats[tls_id]["dur_sum"] += dur
                    tls_stats[tls_id]["count"] += 1
                    break

        rewards = {}
        infos = {}
        for tls_id in self.tls_ids:
            stats = tls_stats[tls_id]
            count = stats["count"]

            avg_wait = stats["wait_sum"] / count if count else 0.0
            reward = float(np.clip(-avg_wait / 300.0, -1.0, 0.0))

            rewards[tls_id] = reward
            infos[tls_id] = [{
                "vehicle_count": count,
                "avg_wait": avg_wait,
                "avg_duration": stats["dur_sum"] / count if count else 0.0
            }]

        return rewards, infos
