import os
from pettingzoo.utils.conversions import aec_to_parallel

import pettingzoo
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch.utils.data import Dataset
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import traci
from lxml import etree
from stable_baselines3.common.callbacks import CheckpointCallback
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from stable_baselines3 import PPO

# === Configuration ===
SUMO_BINARY = "sumo"
SUMO_CONFIG = "osm.sumocfg"
CSV_FILE = "road_load.csv"
ROUTE_FILE = "generated_flows.rou.xml"
STEP_LENGTH = 1
FLOW_DURATION = 30
EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
N_STEPS = 2048
# === Dataset Loader ===
class RouteDataset(Dataset):
    def __init__(self, csv_path):
        print("📥 Loading dataset...")
        self.df = pd.read_csv(csv_path)
        self.edge_vocab = list(pd.concat([self.df["from_edge"], self.df["to_edge"]]).unique())
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edge_vocab)}
        print(f"✅ Loaded {len(self.df)} samples with {len(self.edge_vocab)} unique edges.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        from_edge = self.edge_to_idx[row["from_edge"]]
        to_edge = self.edge_to_idx[row["to_edge"]]
        duration = row["duration_seconds"]
        return from_edge, to_edge, duration, row["from_edge"], row["to_edge"]


class SumoRouteMultiAgentEnv(pettingzoo.AECEnv):
    metadata = {
        "render_modes": [],
        "name": "sumo_multi_route_env_v0",
        "is_parallelizable": True  # ✅ This tells PettingZoo it's safe to convert
    }
    def __init__(self, full_dataset, batch_size=32):
        super().__init__()
        self.render_mode = None
        self.full_dataset = full_dataset
        self.edge_to_idx = full_dataset.edge_to_idx
        self.route_cache = {}
        self.batch_size = batch_size
        self.dataset = None
        self.agents = [f"agent_{i}" for i in range(self.batch_size)]
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            for agent in self.agents
        }

        self.episode_steps = 0
        self.max_episode_steps = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        indices = np.random.choice(len(self.full_dataset), self.batch_size, replace=False)
        self.dataset = [self.full_dataset[i] for i in indices]

        self.obs_dict = {}
        max_edge_idx = max(self.edge_to_idx.values())
        for i, (from_idx, to_idx, duration, *_rest) in enumerate(self.dataset):
            obs = np.array([
                from_idx / max_edge_idx,
                to_idx / max_edge_idx,
                duration / 300.0
            ], dtype=np.float32)
            self.obs_dict[f"agent_{i}"] = obs

        self.agents = self.possible_agents[:]
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.episode_steps = 0
        self.actions = {}


    def observe(self, agent):
        return self.obs_dict[agent]

    def step(self, action):
        agent = self.agent_selection
        self._was_done_step(agent)

        self.actions[agent] = action[0]

        if len(self.actions) < self.batch_size:
            self._cumulative_rewards[agent] = 0.0
            self.agent_selection = self.agents[len(self.actions)]
        else:
            self._run_sumo_simulation()

    def _run_sumo_simulation(self):
        vehicle_counts = (((np.array(list(self.actions.values())) + 1) / 2) * (300 - 1) + 1).astype(int)
        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")
        traci.start([
            SUMO_BINARY,
            "-c", SUMO_CONFIG,
            "-r", ROUTE_FILE,
            "--start",
            "--step-length", str(STEP_LENGTH),
            "--duration-log.statistics"
        ])
        for i, count in enumerate(vehicle_counts):
            from_idx, to_idx, duration, from_edge, to_edge = self.dataset[i]
            route_key = (from_edge, to_edge)

            if route_key not in self.route_cache:
                route = traci.simulation.findRoute(from_edge, to_edge)
                self.route_cache[route_key] = route.edges

            edges = self.route_cache[route_key]
            route_id = f"route_{i}"
            etree.SubElement(root, "route", id=route_id, edges=" ".join(edges))
            etree.SubElement(root, "flow",
                             id=f"{route_id}_flow",
                             type="car",
                             route=route_id,
                             begin="0",
                             end=str(FLOW_DURATION),
                             number=str(count),
                             departPos="random",
                             arrivalPos="random",
                             departSpeed="max",
                             departLane="best")

        etree.ElementTree(root).write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

        vehicle_start_times = {}
        vehicle_end_times = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for veh_id in traci.vehicle.getIDList():
                if veh_id not in vehicle_start_times:
                    vehicle_start_times[veh_id] = traci.simulation.getTime()
                vehicle_end_times[veh_id] = traci.simulation.getTime()
        traci.close()

        # Duration calc
        route_durations = np.zeros(len(self.dataset))
        route_counts = np.zeros(len(self.dataset))
        for veh_id, start_time in vehicle_start_times.items():
            if "_flow." in veh_id:
                parts = veh_id.split("_flow.")[0].split("_")
                if len(parts) >= 2 and parts[0] == "route":
                    route_idx = int(parts[1])
                    if veh_id in vehicle_end_times:
                        sim_dur = vehicle_end_times[veh_id] - start_time
                        route_durations[route_idx] += sim_dur
                        route_counts[route_idx] += 1

        avg_durations = np.divide(route_durations, route_counts, out=np.zeros_like(route_durations),
                                  where=route_counts > 0)
        global_avg = np.mean(avg_durations[route_counts > 0]) if np.any(route_counts > 0) else 9999
        target_avg = np.mean([r[2] for r in self.dataset])
        reward = -abs(global_avg - target_avg) / target_avg

        for agent in self.agents:
            self.rewards[agent] = reward
            self.terminations[agent] = True
            self.truncations[agent] = False
            self.infos[agent] = {}

        self.agents = []
        print(f"Global Avg: {global_avg}, Target Avg: {target_avg}, Reward: {reward}")

    def render(self):
        pass

# === Main Execution ===
dataset = RouteDataset(CSV_FILE)

# === Callback setup ===
checkpoint_callback = CheckpointCallback(
    save_freq=100,
    save_path="./multi_checkpoints/",
    name_prefix="multi_ppo_sumo_traffic"
)


# === Train ===

parallel_env = aec_to_parallel(SumoRouteMultiAgentEnv(dataset, batch_size=64))

vec_env = pettingzoo_env_to_vec_env_v1(parallel_env)

vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1)

# Train SB3 model on VecEnv
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000, callback=checkpoint_callback, progress_bar=True)
