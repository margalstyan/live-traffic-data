import os
import time
import numpy as np
import pandas as pd
import traci
import gymnasium as gym
from gymnasium import spaces
from lxml import etree
from typing import Dict, List


class MultiAgentRouteEnv(gym.Env):
    def __init__(self, sumo_cfg: str, tls_to_routes: Dict[str, List[Dict]], csv_path: str,
                 route_output_path="routes.rou.xml", tripinfo_output_path="tripinfo.xml",
                 sumo_binary="sumo", max_vehicle_per_route=300, max_simulation_time=360):
        super().__init__()
        self.max_simulation_time = max_simulation_time

        self.sumo_cfg = sumo_cfg
        self.csv_path = csv_path
        self.route_output_path = route_output_path
        self.tripinfo_output_path = tripinfo_output_path
        self.sumo_binary = sumo_binary
        self.max_vehicle_per_route = max_vehicle_per_route

        self.tls_to_routes = tls_to_routes
        self.tls_ids = list(tls_to_routes.keys())
        self.raw_data_df = pd.read_csv(self.csv_path)

        self.max_distance = 1000.0
        self.max_duration = 360.0
        self.simulation_running = False

        self.observation_space = {
            tls_id: spaces.Box(low=0, high=1, shape=(len(routes), 4), dtype=np.float32)
            for tls_id, routes in tls_to_routes.items()
        }
        self.action_space = {
            tls_id: spaces.Box(low=0, high=max_vehicle_per_route, shape=(len(routes),), dtype=np.float32)
            for tls_id, routes in tls_to_routes.items()
        }

        self.timestamp_column = None
        self.hour_of_day = 0
        self.episode_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_counter += 1
        self._select_random_timestamp()

        obs = {tls_id: self._build_observation(tls_id) for tls_id in self.tls_ids}
        return obs, {}

    def step(self, actions: Dict[str, np.ndarray]):
        routes_dict = {}
        for tls_id, action_array in actions.items():
            for i, route in enumerate(self.tls_to_routes[tls_id]):
                vehicle_count = int(np.clip(round(action_array[i]), 0, self.max_vehicle_per_route))
                routes_dict[route["id"]] = {
                    "edges": (route["from"], route["to"]),
                    "vehicle_count": vehicle_count
                }

        self._generate_routes_file(routes_dict)

        if self.simulation_running:
            traci.close()
            self.simulation_running = False

        traci.start([
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--route-files", self.route_output_path,
            "--tripinfo-output", self.tripinfo_output_path,
            "--no-step-log", "true",
            "--no-warnings", "true",
        ])
        self.simulation_running = True

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

        sim_time = traci.simulation.getTime()
        traci.close()
        self.simulation_running = False

        # === Parse tripinfo and compute per-agent metrics ===
        route_durations = {}
        route_wait_times = {}
        route_counts = {}

        try:
            tree = etree.parse(self.tripinfo_output_path)
            for trip in tree.getroot().findall("tripinfo"):
                veh_id = trip.attrib["id"]
                route_id = veh_id.split('.')[0]
                duration = float(trip.attrib["duration"])
                wait_time = float(trip.attrib["waitingTime"])

                route_durations.setdefault(route_id, []).append(duration)
                route_wait_times.setdefault(route_id, []).append(wait_time)
                route_counts[route_id] = route_counts.get(route_id, 0) + 1
        except Exception as e:
            print(f"❌ Error parsing tripinfo: {e}")

        # === Rewards + Infos ===
        rewards = {}
        infos = {}

        for tls_id in self.tls_ids:
            routes = self.tls_to_routes[tls_id]
            sim_durations = []
            sim_waits = []
            targets = []
            total_vehicles = 0

            for route in routes:
                route_id = f"route_{route['id']}"
                durs = route_durations.get(route_id, [])
                waits = route_wait_times.get(route_id, [])
                count = route_counts.get(route_id, 0)

                avg_sim = np.mean(durs) if durs else route["target_duration"] * 2
                avg_wait = np.mean(waits) if waits else 0.0

                sim_durations.append(avg_sim)
                sim_waits.append(avg_wait)
                targets.append(route["target_duration"])
                total_vehicles += count

            reward = -np.mean(np.abs(np.array(sim_durations) - np.array(targets)))

            if hasattr(self, "max_simulation_time") and sim_time > self.max_simulation_time:
                excess = (sim_time - self.max_simulation_time) / self.max_simulation_time
                reward += -excess

            rewards[tls_id] = reward
            infos[tls_id] = [{
                "avg_duration": float(np.mean(sim_durations)),
                "avg_wait": float(np.mean(sim_waits)),
                "vehicle_count": total_vehicles
            }]

        obs = {tls_id: self._build_observation(tls_id) for tls_id in self.tls_ids}
        done = {tls_id: True for tls_id in self.tls_ids}
        done["__all__"] = True

        return obs, rewards, done, False, infos

    def _select_random_timestamp(self):
        duration_cols = [col for col in self.raw_data_df.columns if col.startswith("duration_")]
        self.timestamp_column = np.random.choice(duration_cols)
        hhmm = self.timestamp_column.split('_')[-1]
        hour = int(hhmm[:2])
        minute = int(hhmm[2:])
        self.hour_of_day = hour * 60 + minute

        for tls_id in self.tls_ids:
            for i, route in enumerate(self.tls_to_routes[tls_id]):
                idx = route["csv_index"]
                self.tls_to_routes[tls_id][i]["target_duration"] = float(self.raw_data_df.loc[idx, self.timestamp_column])

    def _build_observation(self, tls_id: str) -> np.ndarray:
        hour_norm = self.hour_of_day / 1440.0
        obs = []
        for route in self.tls_to_routes[tls_id]:
            dist = min(route["distance"] / self.max_distance, 1.0)
            free = min(route["duration_without_traffic"] / self.max_duration, 1.0)
            target = min(route["target_duration"] / self.max_duration, 1.0)
            obs.append([dist, free, hour_norm, target])
        return np.array(obs, dtype=np.float32)

    def _generate_routes_file(self, routes_dict):
        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5",
                         maxSpeed="16.6", sigma="0.5")

        route_cache = {}
        FLOW_DURATION = 60
        begin_time = 60

        traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        self.simulation_running = True

        for route_id, data in routes_dict.items():
            if data["vehicle_count"] == 0:
                continue

            from_edge, to_edge = data["edges"]
            key = (from_edge, to_edge)

            if key not in route_cache:
                try:
                    route_result = traci.simulation.findRoute(from_edge, to_edge)
                    if not route_result.edges:
                        print(f"⚠️ No path: {from_edge} → {to_edge}")
                        continue
                    route_cache[key] = route_result.edges
                except Exception as e:
                    print(f"❌ Route error ({route_id}): {e}")
                    continue

            edges = route_cache[key]
            route_uid = f"route_{route_id}"

            etree.SubElement(root, "route", id=route_uid, edges=" ".join(edges))
            etree.SubElement(root, "flow", id=route_uid, type="car", route=route_uid,
                             begin=str(begin_time), end=str(begin_time + FLOW_DURATION),
                             number=str(data["vehicle_count"]), departPos="random", arrivalPos="random")

        traci.close()
        self.simulation_running = False

        tree = etree.ElementTree(root)
        tree.write(self.route_output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def _parse_tripinfo(self):
        route_durations = {}
        try:
            tree = etree.parse(self.tripinfo_output_path)
            for trip in tree.getroot().findall("tripinfo"):
                veh_id = trip.attrib["id"]
                route_id = veh_id.split('.')[0]
                duration = float(trip.attrib["duration"])
                if route_id not in route_durations:
                    route_durations[route_id] = []
                route_durations[route_id].append(duration)
        except Exception as e:
            print(f"❌ Error parsing tripinfo: {e}")
        return route_durations
