import os
import csv
import torch
import torch.nn.functional as F
from policy import RoutePolicy
from utils import (
    RouteDataset,
    scale_action_to_vehicle_count,
    write_flows_and_run_sumo
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 0.01
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

def train():
    dataset = RouteDataset("road_load.csv")
    policy = RoutePolicy().to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # === Load latest model checkpoint if exists ===
    os.makedirs("model", exist_ok=True)
    checkpoint_files = [f for f in os.listdir("model") if f.startswith("route_policy_epoch_") and f.endswith(".pt")]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))  # sort by epoch
        latest = checkpoint_files[-1]
        policy.load_state_dict(torch.load(f"model/{latest}"))
        print(f"🔁 Loaded checkpoint from: model/{latest}")

    route_cache = {}

    for epoch in range(1000):
        batch = dataset.sample_batch(batch_size=BATCH_SIZE)
        max_edge_idx = max(dataset.edge_to_idx.values())

        obs = torch.tensor([[
                from_idx / max_edge_idx,
                to_idx / max_edge_idx,
                duration / 300.0,
                distance / 3000.0
            ] for from_idx, to_idx, duration, distance, *_ in batch],
            dtype=torch.float32
        ).to(DEVICE)

        actions = policy(obs).squeeze()
        vehicle_counts = scale_action_to_vehicle_count(actions)

        total_durations = write_flows_and_run_sumo(batch, vehicle_counts, route_cache)
        target_duration = torch.tensor([b[2] for b in batch], dtype=torch.float32)

        mse = F.mse_loss(total_durations, target_duration)
        loss = mse * actions.mean().abs() * SCALE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for curr, targ in zip(total_durations, target_duration):
            print(f"[Epoch {epoch}] Avg Duration: {curr:.2f} | Target: {targ:.2f}")
        print(f"MSE: {mse:.4f} | Loss: {loss.item():.4f}")

        if epoch % 10 == 0:
            torch.save(policy.state_dict(), f"model/route_policy_epoch_{epoch}.pt")
            print(f"✅ Model checkpoint saved at epoch {epoch}")

        with open("loss_mse_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss.item(), mse.item()])


if __name__ == "__main__":
    train()
