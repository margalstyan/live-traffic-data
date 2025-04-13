from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch.utils.data import Dataset
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import traci
from lxml import etree
import os

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


class SumoTrafficMultiRouteEnv(gym.Env):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.route_cache = {}

        self.observation_space = spaces.Box(
            low=0, high=1e3, shape=(len(self.dataset), 3), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.dataset),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = np.zeros((len(self.dataset), 3), dtype=np.float32)
        max_edge_idx = max(self.dataset.edge_to_idx.values())

        for i in range(len(self.dataset)):
            from_idx, to_idx, duration, *_ = self.dataset[i]
            self.obs[i] = np.array([
                from_idx / max_edge_idx,
                to_idx / max_edge_idx,
                duration / 300
            ], dtype=np.float32)
        return self.obs, {}

    def step(self, actions):
        vehicle_counts = (((actions + 1) / 2) * (300 - 1) + 1).astype(int)
        print(f"🚦 Generating flows for {len(self.dataset)} routes")

        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

        for i, count in enumerate(vehicle_counts):
            from_idx, to_idx, duration, from_edge, to_edge = self.dataset[i]
            route_key = (from_edge, to_edge)

            if route_key not in self.route_cache:
                traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])
                route = traci.simulation.findRoute(from_edge, to_edge)
                self.route_cache[route_key] = route.edges
                traci.close()

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
        print(f"✅ {ROUTE_FILE} written")

        traci.start([
            SUMO_BINARY,
            "-c", SUMO_CONFIG,
            "-r", ROUTE_FILE,
            "--start",
            "--step-length", str(STEP_LENGTH),
            "--duration-log.statistics"
        ])

        vehicle_start_times = {}
        vehicle_end_times = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for veh_id in traci.vehicle.getIDList():
                if veh_id not in vehicle_start_times:
                    vehicle_start_times[veh_id] = traci.simulation.getTime()
                vehicle_end_times[veh_id] = traci.simulation.getTime()
        traci.close()

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
        rewards = np.zeros(len(self.dataset))
        target_durations = np.array([r[2] for r in self.dataset])
        for i in range(len(self.dataset)):
            if route_counts[i] > 0:
                rewards[i] = -abs(avg_durations[i] - target_durations[i]) / target_durations[i]
            else:
                rewards[i] = -1.0

        avg_loss = np.mean(np.abs(avg_durations - target_durations))

        # ⬇️ Log target vs simulated duration per route
        duration_logs = []
        for i in range(len(self.dataset)):
            duration_logs.append({
                "route": f"route_{i}",
                "target_duration": float(target_durations[i]),
                "simulated_duration": float(avg_durations[i]),
                "vehicles": int(route_counts[i]),
                "reward": float(rewards[i])
            })

        print("🛣 Route duration summary:")
        for log in duration_logs:
            print(f"  {log['route']}: "
                  f"Target={log['target_duration']:.1f}s, "
                  f"Sim={log['simulated_duration']:.1f}s, "
                  f"Vehicles={log['vehicles']}, "
                  f"Reward={log['reward']:.3f}")

        terminated = True
        truncated = False
        info = {
            "avg_loss": avg_loss,
            "duration_logs": duration_logs
        }
        print(f"📊 Avg loss: {avg_loss:.3f} | "
                f"Avg reward: {np.mean(rewards):.3f} | "
                f"Avg vehicles: {np.mean(route_counts):.2f}")

        return self.obs, rewards.mean(), terminated, truncated, info


# === CONFIGURATION ===
SUMO_BINARY = "sumo"
SUMO_CONFIG = "osm.sumocfg"
CSV_FILE = "road_load.csv"
ROUTE_FILE = "generated_flows.rou.xml"
STEP_LENGTH = 1
FLOW_DURATION = 30
EPOCHS = 10
LEARNING_RATE = 1e-3
SYNC_WEIGHT = 10

# === Dataset & Env ===
dataset = RouteDataset(CSV_FILE)
env = SumoTrafficMultiRouteEnv(dataset)
check_env(env, warn=True)

# === Model ===
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# === Logging Training Progress ===
# reward_log = []
# loss_log = []
#
# for episode in range(100):  # You can increase this number
#     obs, _ = env.reset()
#     action, _ = model.predict(obs)
#     obs, reward, done, _, info = env.step(action)
#
#     reward_log.append(reward)
#     loss_log.append(info["avg_loss"])
#     print(f"📈 EP {episode:03} — Reward: {reward:.3f} | Avg loss: {info['avg_loss']:.2f}")

# === (Optional) Save Logs to CSV ===
# pd.DataFrame({
#     "episode": list(range(len(reward_log))),
#     "reward": reward_log,
#     "avg_loss": loss_log
# }).to_csv("training_log.csv", index=False)
# print("✅ Training log saved to training_log.csv")
