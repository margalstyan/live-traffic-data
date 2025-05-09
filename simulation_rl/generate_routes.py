import os
import numpy as np
import pandas as pd
from lxml import etree  # Updated to use lxml for pretty XML
from stable_baselines3 import SAC
import random


def load_routes(csv_path: str, timestamp_column: str) -> list:
    df = pd.read_csv(csv_path)
    routes = []
    for i, row in df.iterrows():
        route = {
            "id": i,
            "edges": [str(row["from_edge"]), str(row["to_edge"])],
            "target_duration": float(row[timestamp_column]),
            "duration_without_traffic": float(row["duration_without_traffic"]),
            "distance": float(row["distance"])
        }
        routes.append(route)
    return routes


def build_observations(routes, timestamp_str, max_dist=1000.0, max_dur=300.0):
    hhmm = timestamp_str.split('_')[-1]
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    minutes_of_day = hour * 60 + minute
    hour_norm = minutes_of_day / 1440.0

    obs = []
    for r in routes:
        dist = min(r["distance"] / max_dist, 1.0)
        free = min(r["duration_without_traffic"] / max_dur, 1.0)
        obs.append([dist, free, hour_norm])
    return np.array(obs, dtype=np.float32)


def generate_route_file(routes, actions, output_file="predicted_routes.rou.xml"):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    for i, r in enumerate(routes):
        flow_id = f"flow_route_{r['id']}"
        from_edge = r["edges"][0]
        to_edge = r["edges"][-1]

        etree.SubElement(
            root, "flow",
            attrib={
                "id": flow_id,
                "type": "car",
                "from": from_edge,
                "to": to_edge,
                "begin": "0",
                "end": "60",
                "number": str(actions[i]),
                "departPos": "random",
                "arrivalPos": "random"
            }
        )

    tree = etree.ElementTree(root)
    tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"✅ Route file written to {output_file}")


def generate_routes_with_sac_model(
    model_path: str = "./sac_checkpoints/1/sac1_model_1000_steps.zip",
    csv_path: str = "../data/final_with_all_data.csv",
    output_file: str = "routes_for_training.rou.xml"
):
    # Load and sample timestamp column
    df = pd.read_csv(csv_path)
    candidate_columns = [col for col in df.columns if col.startswith("duration_") and not col.endswith("without_traffic")]
    if not candidate_columns:
        raise ValueError("❌ No timestamp columns found in CSV.")
    timestamp = random.choice(candidate_columns)
    print(f"📅 Selected timestamp column: {timestamp}")

    # Load SAC model
    model = SAC.load(model_path)

    # Load routes and build observations
    routes = load_routes(csv_path, timestamp)
    obs = build_observations(routes, timestamp)

    # Predict actions
    actions, _ = model.predict(obs, deterministic=True)
    actions = np.clip(np.round(actions), 0, 300).astype(int)

    # Generate route file
    generate_route_file(routes, actions, output_file)
