import pandas as pd
import networkx as nx
from collections import deque

def build_combined_routes_with_edges(
    mixed_data_file="mixed_route_data.csv",
    output_file="combined_routes_with_edges.csv",
    min_len=4,
    max_len=10
):
    print("[*] Loading data...")
    df = pd.read_csv(mixed_data_file)
    duration_cols = [col for col in df.columns if col.startswith("duration_")]

    # Build the graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row["Origin"], row["Destination"],
            direction=row["Direction"],
            distance=row["distance"],
            from_edge=row["from_edge"],
            to_edge=row["to_edge"],
            durations=row[duration_cols].values.astype(float),
            avg_speed=row["avg_speed"]
        )

    print("[*] Generating paths...")
    all_paths = []
    for start_node in G.nodes:
        queue = deque()
        queue.append((start_node, [start_node]))
        while queue:
            current, path = queue.popleft()
            if len(path) > max_len:
                continue
            if min_len <= len(path) - 1 <= max_len:
                all_paths.append(path)
            for neighbor in sorted(G.successors(current), key=lambda n: G[current][n]["avg_speed"]):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

    print(f"[✓] Found {len(all_paths)} valid paths.")

    # Build the combined route records
    combined_routes = []
    for path in all_paths:
        total_distance = 0
        total_durations = [0] * len(duration_cols)
        edge_ids = []

        for i in range(len(path) - 1):
            edge = G[path[i]][path[i + 1]]
            total_distance += edge["distance"]
            total_durations = [a + b for a, b in zip(total_durations, edge["durations"])]
            edge_ids.append(edge["from_edge"])

        # Add last to_edge from final segment
        edge_ids.append(edge["to_edge"])

        combined_routes.append({
            "route": " → ".join(path),
            "num_segments": len(path) - 1,
            "total_distance": total_distance,
            "edge_list": " ".join(edge_ids),
            **{col: total_durations[i] for i, col in enumerate(duration_cols)}
        })

    output_df = pd.DataFrame(combined_routes)
    output_df.to_csv(output_file, index=False)
    print(f"[✓] Saved {len(output_df)} combined routes to {output_file}")

# Run the script
if __name__ == "__main__":
    build_combined_routes_with_edges()
