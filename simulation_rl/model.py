import gymnasium as gym
import numpy as np
import traci
import os
from gymnasium import spaces
from lxml import etree
import xml.etree.ElementTree as ET
from collections import defaultdict


class SumoTrafficEnv(gym.Env):
    def __init__(self, routes, sumo_config, sumo_binary="sumo", max_vehicles=200):
        super(SumoTrafficEnv, self).__init__()

        self.routes = routes  # dictionary of {route_id: {from_edge, to_edge, target_duration, duration_without_traffic}}
        self.route_ids = list(routes.keys())
        self.sumo_config = sumo_config
        self.sumo_binary = sumo_binary
        self.max_vehicles = max_vehicles

        self.num_routes = len(self.routes)

        # --- Gym spaces ---
        # Actions: number of vehicles per route (continuous [0, max_vehicles])
        self.action_space = spaces.Box(low=0, high=self.max_vehicles, shape=(self.num_routes,), dtype=np.float32)

        # Observations: duration_without_traffic and target_duration per route
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_routes * 2,), dtype=np.float32)

        # Constants
        self.FLOW_DURATION = 60  # seconds
        self.ROUTE_FILE = "config/generated_rl_flows.rou.xml"
        self.TRIPINFO_FILE = "output/tripinfo_rl.xml"

        # Prepare route cache
        self.route_cache = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = []
        for route_id in self.route_ids:
            info = self.routes[route_id]
            obs.append(info["duration_without_traffic"])
            obs.append(info["target_duration"])

        info = {}
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        vehicle_counts = np.clip(action, 0, self.max_vehicles)
        vehicle_counts = vehicle_counts.astype(int)

        self._generate_flow_file(vehicle_counts)
        self._run_sumo_simulation()
        mean_durations = self._parse_tripinfos()

        # Calculate success rate reward
        duration_errors = []
        successes = 0
        total = 0

        for route_id in self.route_ids:
            expected = self.routes[route_id]["target_duration"]
            simulated = mean_durations.get(route_id, expected)
            if expected > 0 and simulated > 0:
                relative_error = abs(simulated - expected) / max(simulated, expected)
                duration_errors.append(relative_error)
                if relative_error <= 0.2:
                    successes += 1
                total += 1

        reward = successes / total if total > 0 else 0.0  # Success rate
        reward = reward * 10.0  # Optional: amplify reward scale

        mean_duration_error = np.mean(duration_errors) if duration_errors else 1.0

        next_obs, _ = self.reset()

        terminated = True
        truncated = False

        info = {
            "mean_durations": mean_durations,
            "vehicle_counts": vehicle_counts.tolist(),
            "mean_duration_error": mean_duration_error
        }

        return next_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass  # No rendering needed for now

    def _generate_flow_file(self, vehicle_counts):
        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

        if traci.isLoaded():
            traci.close()

        # Start SUMO just for checking connections
        traci.start([self.sumo_binary, "-c", self.sumo_config, "--start", "--step-length", "1",
                 "--no-warnings", "true", "--log", os.devnull],
                    label="connection_check")

        for idx, route_id in enumerate(self.route_ids):
            info = self.routes[route_id]
            count = vehicle_counts[idx]
            if count <= 0:
                continue

            try:
                route = traci.simulation.findRoute(info["from_edge"], info["to_edge"])
                if len(route.edges) == 0:
                    print(f"⚠️ No valid path for {info['from_edge']} -> {info['to_edge']}. Skipping flow.")
                    continue
            except Exception:
                print(f"⚠️ Route check failed for {info['from_edge']} -> {info['to_edge']}. Skipping flow.")
                continue

            flow = etree.SubElement(root, "flow")
            flow.set("id", route_id)
            flow.set("type", "car")
            flow.set("from", info["from_edge"])
            flow.set("to", info["to_edge"])
            flow.set("begin", "0")
            flow.set("end", str(self.FLOW_DURATION))
            flow.set("number", str(count))
            flow.set("departPos", "random")
            flow.set("arrivalPos", "random")

        traci.close()  # Close connection checker

        tree = etree.ElementTree(root)
        os.makedirs(os.path.dirname(self.ROUTE_FILE), exist_ok=True)
        tree.write(self.ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def _run_sumo_simulation(self):
        if traci.isLoaded():
            traci.close()

        traci.start([self.sumo_binary, "-c", self.sumo_config, "-r", self.ROUTE_FILE,
                     "--tripinfo-output", self.TRIPINFO_FILE, "--start", "--step-length", str(1),
                 "--no-warnings", "true", "--log", os.devnull])

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

        traci.close()

    def _parse_tripinfos(self):
        if not os.path.exists(self.TRIPINFO_FILE):
            print(f"⚠️ Warning: {self.TRIPINFO_FILE} not found.")
            return {}

        tree = ET.parse(self.TRIPINFO_FILE)
        root = tree.getroot()

        route_durations = defaultdict(list)

        for tripinfo in root.findall('tripinfo'):
            trip_id = tripinfo.attrib['id']
            duration = float(tripinfo.attrib['duration'])
            base_route = trip_id.split('.')[0]
            route_durations[base_route].append(duration)

        mean_durations = {route: sum(durations) / len(durations) for route, durations in route_durations.items()}
        return mean_durations

    def _calculate_reward(self, mean_durations):
        total_error = 0

        for route_id in self.route_ids:
            expected = self.routes[route_id]["target_duration"]
            simulated = mean_durations.get(route_id, expected)  # fallback to expected if missing (no vehicles)

            error = abs(simulated - expected)
            total_error += error

        reward = -total_error
        return reward
