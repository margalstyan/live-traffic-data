from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch.utils.data import Dataset
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import traci
from lxml import etree

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


class SumoTrafficEnv(gym.Env):
    def __init__(self, dataset, ):
        super(SumoTrafficEnv, self).__init__()
        self.dataset = dataset
        self.index = 0  # current route index
        self.route_cache = {}

        self.observation_space = spaces.Box(low=0, high=1e3, shape=(1 + 2,), dtype=np.float32)  # duration + 2 edge indices
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.index >= len(self.dataset):
            self.index = 0
        self.from_idx, self.to_idx, self.target_duration, self.from_edge, self.to_edge = self.dataset[self.index]
        self.index += 1
        max_edge_idx = max(self.dataset.edge_to_idx.values())

        obs = np.array([
            self.from_idx / max_edge_idx,
            self.to_idx / max_edge_idx,
            self.target_duration / 300
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        print(f"🚦 [STEP] Action received: {action}")
        raw_action = np.clip(action[0], -1, 1)
        veh_count = ((raw_action + 1) / 2) * (300 - 1) + 1

        print(f"\n🚦 [STEP] Starting simulation for route {self.index}/{len(self.dataset)}")
        print(f"➡️ From: {self.from_edge} | To: {self.to_edge}")
        print(f"🚗 Action (vehicle count): {veh_count:.2f}")

        route_key = (self.from_edge, self.to_edge)
        if route_key not in self.route_cache:
            print("📡 Connecting to SUMO to fetch route path...")
            traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])
            route = traci.simulation.findRoute(self.from_edge, self.to_edge)
            self.route_cache[route_key] = route.edges
            traci.close()
            print(f"✅ Route cached for: {self.from_edge} → {self.to_edge}")

        edges = self.route_cache[route_key]

        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")
        etree.SubElement(root, "route", id="route0", edges=" ".join(edges))
        etree.SubElement(root, "flow", id="route0", type="car", route="route0",
                         begin="0", end=str(FLOW_DURATION),
                         number=str(int(veh_count)),
                         departPos="random", arrivalPos="random")
        etree.ElementTree(root).write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE, "--start", "--step-length", str(STEP_LENGTH)])
        vehicle_data = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for veh_id in traci.vehicle.getIDList():
                if veh_id not in vehicle_data:
                    vehicle_data[veh_id] = traci.simulation.getTime()
        traci.close()

        sim_duration = max(vehicle_data.values()) - min(vehicle_data.values()) if vehicle_data else 0.0
        reward = -abs(sim_duration - self.target_duration) / self.target_duration

        print(f"🕒 Simulated duration: {sim_duration:.2f}s")
        print(f"🎯 Target duration:    {self.target_duration:.2f}s")
        print(f"🏆 Reward:             {reward:.2f}")

        obs = np.array([self.from_idx, self.to_idx, self.target_duration], dtype=np.float32)
        terminated = True
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "osm.sumocfg"
CSV_FILE = "road_load.csv"
ROUTE_FILE = "generated_flows.rou.xml"
STEP_LENGTH = 1
FLOW_DURATION = 30
EPOCHS = 10
LEARNING_RATE = 1e-3
SYNC_WEIGHT = 10


dataset = RouteDataset(CSV_FILE)

env = SumoTrafficEnv(dataset)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE)
model.learn(total_timesteps=50000)
