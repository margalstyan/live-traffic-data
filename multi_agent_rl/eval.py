import numpy as np
import traci
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import xml.etree.ElementTree as ET
from train_lstm import MultiTLEnvSingleAgent  # Assumes your custom env class is in train.py

# === Configuration ===
NET_FILE = str(Path("../config/osm.net.xml").resolve())
ROUTE_FILE = str(Path("../config/routes.rou.xml").resolve())
MODEL_FILE = "./checkpoints_lstm/4/ppo_tl_model_lstm_30000_steps.zip"
OUTPUT_XML_FILE = "predicted_tls.xml"

# === Create dummy env to load the model ===
dummy_env = MultiTLEnvSingleAgent(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=False,
    num_seconds=2200,
    yellow_time=3,
    min_green=5,
    max_green=90,
    fixed_ts=True,
)

# === Load PPO model ===
model = RecurrentPPO.load(MODEL_FILE, env=dummy_env)

# === Create real env with model passed in ===
env = MultiTLEnvSingleAgent(
    model=model,
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=False,
    num_seconds=2200,
    yellow_time=3,
    min_green=5,
    max_green=90,
    fixed_ts=True,
)

# === Reset env and predict ===
obs, _ = env.reset()
predicted_action, _ = model.predict(obs, deterministic=True)
scaled_actions = env.min_duration + (predicted_action + 1.0) * (env.max_duration - env.min_duration) / 2.0

# === Build output XML ===
tls_element = ET.Element('additional')
idx = 0

for ts_id in env.ts_ids:
    logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
    phases = logic.phases

    for phase_idx in env.phase_indices_per_tl[ts_id]:
        predicted_duration = float(np.clip(scaled_actions[idx], env.min_duration, env.max_duration))
        phases[phase_idx].duration = predicted_duration
        idx += 1

    # Write to XML
    tl_logic = ET.SubElement(
        tls_element, 'tlLogic',
        attrib={
            'id': ts_id,
            'programID': 'predicted',
            'offset': '0',
            'type': 'static'
        }
    )

    for phase in phases:
        ET.SubElement(
            tl_logic, 'phase',
            attrib={
                'duration': str(phase.duration),
                'state': phase.state
            }
        )

# === Save to file ===
tree = ET.ElementTree(tls_element)
ET.indent(tree, space="  ")
tree.write(OUTPUT_XML_FILE, encoding="utf-8", xml_declaration=True)

print(f"✅ Predicted TL phases successfully written to {OUTPUT_XML_FILE}")
