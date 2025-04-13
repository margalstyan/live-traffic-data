import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import traci
from lxml import etree

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

# === Neural Network ===
class VehicleFlowNN(nn.Module):
    def __init__(self, num_edges, embedding_dim=16):
        super(VehicleFlowNN, self).__init__()
        print("🧠 Initializing VehicleFlowNN model...")
        self.from_embedding = nn.Embedding(num_edges, embedding_dim)
        self.to_embedding = nn.Embedding(num_edges, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        print("✅ Model ready.")

    def forward(self, from_edge, to_edge, duration):
        from_emb = self.from_embedding(from_edge)
        to_emb = self.to_embedding(to_edge)
        x = torch.cat([from_emb, to_emb, duration.unsqueeze(1)], dim=1)
        return self.fc_layers(x)

# === Generate Multi-Flow XML
def generate_multi_flow(dataset, model, route_cache):
    print("🔄 Generating multi-route flow XML...")
    model.eval()
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    for i in range(len(dataset)):
        from_idx, to_idx, duration, from_edge, to_edge = dataset[i]
        from_idx = torch.tensor([from_idx])
        to_idx = torch.tensor([to_idx])
        duration_tensor = torch.tensor([duration], dtype=torch.float32)

        with torch.no_grad():
            vehicle_count = model(from_idx, to_idx, duration_tensor).item()
        vehicle_count = max(10, min(vehicle_count, 1000))

        route_id = f"route{i}"
        if (from_edge, to_edge) not in route_cache:
            route = traci.simulation.findRoute(from_edge, to_edge)
            route_cache[(from_edge, to_edge)] = route.edges
        edges = route_cache[(from_edge, to_edge)]

        etree.SubElement(root, "route", id=route_id, edges=" ".join(edges))
        etree.SubElement(root, "flow", id=route_id, type="car", route=route_id,
                         begin="0", end=str(FLOW_DURATION),
                         vehsPerHour=str(int(vehicle_count * 3600 // FLOW_DURATION)),
                         departPos="random", arrivalPos="random")

    etree.ElementTree(root).write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print("✅ XML generation complete.")

# === Training Loop ===
dataset = RouteDataset(CSV_FILE)
model = VehicleFlowNN(num_edges=len(dataset.edge_vocab))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
route_cache = {}

for epoch in range(EPOCHS):
    print(f"\n🧪 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0.0

    for i in range(len(dataset)):
        from_idx, to_idx, target_duration, from_edge, to_edge = dataset[i]
        from_idx = torch.tensor([from_idx])
        to_idx = torch.tensor([to_idx])
        duration_tensor = torch.tensor([target_duration], dtype=torch.float32)
        target_duration_tensor = torch.tensor([target_duration], dtype=torch.float32)

        predicted_count = model(from_idx, to_idx, duration_tensor).item()
        predicted_count = max(10, min(predicted_count, 1000))
        print(f"🚗 Route {i+1}: {from_edge} → {to_edge} | Predicted vehicles/hour: {predicted_count:.2f}")

        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])
        root = etree.Element("routes")
        etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

        if (from_edge, to_edge) not in route_cache:
            route = traci.simulation.findRoute(from_edge, to_edge)
            route_cache[(from_edge, to_edge)] = route.edges
        edges = route_cache[(from_edge, to_edge)]

        etree.SubElement(root, "route", id="route0", edges=" ".join(edges))
        etree.SubElement(root, "flow", id="route0", type="car", route="route0",
                         begin="0", end=str(FLOW_DURATION),
                         vehsPerHour=str(int(predicted_count * 3600 // FLOW_DURATION)),
                         departPos="random", arrivalPos="random")
        etree.ElementTree(root).write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        traci.close()

        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE, "--start", "--step-length", str(STEP_LENGTH)])
        vehicle_data = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for veh_id in traci.vehicle.getIDList():
                if veh_id not in vehicle_data:
                    vehicle_data[veh_id] = traci.simulation.getTime()
        traci.close()

        sim_duration = max(vehicle_data.values()) - min(vehicle_data.values()) if vehicle_data else 0.0
        sim_duration_tensor = torch.tensor([sim_duration], dtype=torch.float32)

        loss = loss_fn(sim_duration_tensor, target_duration_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"🧮 Loss: {loss.item():.4f} | Simulated: {sim_duration:.2f}s vs Target: {target_duration:.2f}s")

    # === Synchronized Flow Simulation
    print("🌐 Running synchronized flow simulation...")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])
    generate_multi_flow(dataset, model, route_cache)
    traci.close()

    model.eval()
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE, "--start", "--step-length", str(STEP_LENGTH)])
    vehicle_data = {}
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in vehicle_data:
                vehicle_data[veh_id] = {
                    "start": traci.simulation.getTime(),
                    "route_id": traci.vehicle.getRouteID(veh_id)
                }
        for veh_id in list(vehicle_data):
            if veh_id not in traci.vehicle.getIDList() and vehicle_data[veh_id].get("end") is None:
                vehicle_data[veh_id]["end"] = traci.simulation.getTime()
    traci.close()

    durations_by_route = {}
    for veh_id, data in vehicle_data.items():
        if "start" in data and "end" in data:
            rid = data["route_id"]
            dur = data["end"] - data["start"]
            durations_by_route.setdefault(rid, []).append(dur)

    sync_loss = 0
    for i in range(len(dataset)):
        route_id = f"route{i}"
        if route_id in durations_by_route and durations_by_route[route_id]:
            avg_duration = sum(durations_by_route[route_id]) / len(durations_by_route[route_id])
            target_duration = dataset[i][2]
            sync_loss += (avg_duration - target_duration) ** 2

    weighted_sync_loss = SYNC_WEIGHT * sync_loss
    print(f"📊 Epoch {epoch+1} Summary:")
    print(f"   - Real Duration Loss: {total_loss:.2f}")
    print(f"   - Sync Loss (weighted): {weighted_sync_loss:.2f}")
    print(f"   ✅ Total Epoch Loss: {total_loss + weighted_sync_loss:.2f}")
