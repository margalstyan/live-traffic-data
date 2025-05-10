import pandas as pd
import traci
from lxml import etree
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import xml.etree.ElementTree as ET


class MultiRouteSUMOGymEnv(gym.Env):
    def __init__(self, sumo_config_path, route_data_list,
                 route_file_path=None, tripinfo_path=None,
                 sumo_gui=False, csv_path=None, instance_id=0):
        super().__init__()
        self.max_simulation_time = 360
        self.sumo_binary = "sumo-gui" if sumo_gui else "sumo"
        self.sumo_config = sumo_config_path
        self.routes = route_data_list
        self.csv_path = csv_path

        self.instance_id = instance_id
        self.route_file_path = route_file_path or f"routes_{instance_id}.rou.xml"
        self.tripinfo_path = tripinfo_path or f"tripinfo_{instance_id}.xml"

        self.num_routes = len(self.routes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_routes, 4), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=300, shape=(self.num_routes,), dtype=np.float32)

        self.max_distance = 1000.0
        self.max_duration = 360.0

        self.raw_data_df = None
        self.timestamp_column = None
        self.hour_of_day = 0
        self.simulation_running = False
        self.last_observation = None
        self.episode_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_counter += 1

        if self.csv_path:
            self.pick_random_timestamp_column(self.csv_path)

        hour_norm = self.hour_of_day / 1440.0
        obs = []
        for route in self.routes:
            dist = min(route["distance"] / self.max_distance, 1.0)
            free = min(route["duration_without_traffic"] / self.max_duration, 1.0)
            target = min(route["target_duration"] / self.max_duration, 1.0)
            obs.append([dist, free, hour_norm, target])

        self.last_observation = np.array(obs, dtype=np.float32)
        return self.last_observation.copy(), {}

    def step(self, action):
        actions = np.clip(np.round(action), 0, 300).astype(int)

        routes_dict = {
            route["id"]: {
                "edges": route["edges"],
                "vehicle_count": actions[idx]
            }
            for idx, route in enumerate(self.routes)
        }

        self._generate_routes_file(routes_dict)

        if self.simulation_running:
            traci.close()
            self.simulation_running = False

        # Dynamically adjust step length
        step_len = max(1, 5 - self.episode_counter // 500)

        traci.start([
            self.sumo_binary,
            "-c", self.sumo_config,
            "--route-files", self.route_file_path,
            "--tripinfo-output", self.tripinfo_path,
            "--step-length", str(step_len),
            "--no-step-log", "true",
            "--no-warnings", "true",
        ])
        self.simulation_running = True

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

        sim_time = traci.simulation.getTime()
        traci.close()
        self.simulation_running = False

        # === Collect durations ===
        route_durations = {f"route_{r['id']}": [] for r in self.routes}
        try:
            tree = ET.parse(self.tripinfo_path)
            for trip in tree.getroot().findall("tripinfo"):
                veh_id = trip.attrib["id"]
                route_id = veh_id.split('.')[0]
                duration = float(trip.attrib["duration"])
                if route_id in route_durations:
                    route_durations[route_id].append(duration)
        except Exception as e:
            print(f"[ENV {self.instance_id}] ❌ Error reading tripinfo: {e}")

        # === Compute reward ===
        simulated_durations = []
        for route in self.routes:
            r_id = f"route_{route['id']}"
            values = route_durations.get(r_id, [])
            avg_sim = np.mean(values) if values else route["target_duration"] * 2
            simulated_durations.append(avg_sim)

        targets = np.array([r["target_duration"] for r in self.routes])
        base_reward = -np.mean((np.array(simulated_durations) - targets) ** 2)

        penalty = 0.0
        if sim_time > self.max_simulation_time:
            excess_ratio = (sim_time - self.max_simulation_time) / self.max_simulation_time
            penalty = -base_reward * excess_ratio

        reward = base_reward + penalty
        print(f"[ENV {self.instance_id}] Reward: {reward:.4f}, Step length: {step_len}, Sim time: {sim_time:.2f}")

        return self.last_observation.copy(), reward, True, False, {}

    def _generate_routes_file(self, routes_dict):
        if not traci.isLoaded():
            traci.start([self.sumo_binary, "-c", self.sumo_config])
            self.simulation_running = True

        route_cache = {}
        FLOW_DURATION = 60
        begin_time = 60

        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5",
                         maxSpeed="16.6", sigma="0.5")

        for route_id, data in routes_dict.items():
            if data["vehicle_count"] == 0:
                continue

            from_edge, to_edge = data["edges"]
            key = (from_edge, to_edge)

            if key not in route_cache:
                try:
                    route_result = traci.simulation.findRoute(from_edge, to_edge)
                    if not route_result.edges:
                        print(f"[ENV {self.instance_id}] ⚠️ No path: {from_edge} → {to_edge}")
                        continue
                    route_cache[key] = route_result.edges
                except Exception as e:
                    print(f"[ENV {self.instance_id}] ❌ Route error ({route_id}): {e}")
                    continue

            edges = route_cache[key]
            route_uid = f"route_{route_id}"

            etree.SubElement(root, "route", id=route_uid, edges=" ".join(edges))
            etree.SubElement(root, "flow", id=route_uid, type="car", route=route_uid,
                             begin=str(begin_time), end=str(begin_time + FLOW_DURATION),
                             number=str(data["vehicle_count"]), departPos="random", arrivalPos="random")

        tree = etree.ElementTree(root)
        tree.write(self.route_file_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def pick_random_timestamp_column(self, csv_path: str):
        if self.raw_data_df is None:
            self.raw_data_df = pd.read_csv(csv_path)

        duration_cols = [col for col in self.raw_data_df.columns if col.startswith("duration_2025")]
        self.timestamp_column = np.random.choice(duration_cols)

        hhmm = self.timestamp_column.split('_')[-1]
        hour = int(hhmm[:2])
        minute = int(hhmm[2:])
        self.hour_of_day = hour * 60 + minute

        for i, route in enumerate(self.routes):
            self.routes[i]["target_duration"] = float(self.raw_data_df.loc[i, self.timestamp_column])
