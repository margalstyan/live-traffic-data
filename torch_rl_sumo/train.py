import torch
import torch.distributions as dist
import concurrent.futures
from policy import RoutePolicy
from utils import (
    RouteDataset,
    scale_action_to_vehicle_count,
    write_flows_and_run_sumo
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

NUM_ENVS = 10
BATCH_SIZE = 32

def simulate_one(args):
    sub_batch, sub_counts, route_cache, run_id = args
    return write_flows_and_run_sumo(sub_batch, sub_counts, route_cache, run_id=run_id)


def train():
    dataset = RouteDataset("road_load.csv")
    policy = RoutePolicy(input_dim=4).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    route_cache = {}

    for epoch in range(1000):
        batch = dataset.sample_batch(batch_size=BATCH_SIZE)
        max_edge_idx = max(dataset.edge_to_idx.values())

        obs = torch.tensor([
            [from_idx / max_edge_idx, to_idx / max_edge_idx, duration / 300.0, i / BATCH_SIZE]
            for i, (from_idx, to_idx, duration, *_rest) in enumerate(batch)
        ], dtype=torch.float32).to(DEVICE)

        mean, std = policy(obs)
        normal = dist.Normal(mean, torch.clamp(std, min=1e-3))
        actions = normal.rsample()
        log_probs = normal.log_prob(actions)

        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        vehicle_counts = scale_action_to_vehicle_count(clipped_actions)

        # Prepare args for each parallel process
        sub_batches = []
        for i in range(NUM_ENVS):
            indices = list(range(i, BATCH_SIZE, NUM_ENVS))
            b = [batch[j] for j in indices]
            c = vehicle_counts[indices].cpu()
            sub_batches.append((b, c, route_cache, i))

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_ENVS) as executor:
            results = list(executor.map(simulate_one, sub_batches))

        durations = [d for group in results for d in group]

        target_durations = torch.tensor([r[2] for r in batch], dtype=torch.float32).to(DEVICE)
        durations_tensor = torch.tensor(durations, dtype=torch.float32).to(DEVICE)

        rewards = -torch.tanh((durations_tensor - target_durations) / target_durations)
        mask = durations_tensor != 9999

        if mask.sum() > 0:
            valid_rewards = rewards[mask]
            valid_log_probs = log_probs[mask]
            baseline = valid_rewards.mean()
            advantages = valid_rewards - baseline
            loss = -torch.mean(advantages * valid_log_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Avg Reward: {valid_rewards.mean().item():.4f}")

            #save loss and rewards
            with open("loss_rewards.txt", "a") as f:
                f.write(f"{epoch},{loss.item()},{valid_rewards.mean().item()}\n")
        else:
            print(f"[Epoch {epoch}] Skipped update due to all failed routes")

        # Save model every 100 epochs
        if epoch % 10 == 0:
            torch.save(policy.state_dict(), f"model/policy_epoch_{epoch}.pth")
            print(f"Model saved at epoch {epoch}")


if __name__ == "__main__":
    train()
