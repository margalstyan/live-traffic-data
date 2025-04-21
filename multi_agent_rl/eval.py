import numpy as np
import traci
from pathlib import Path
from sumo_rl.environment.env import SumoEnvironment
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import xml.etree.ElementTree as ET

from train import MultiTLEnvSingleAgent

# Set paths explicitly
NET_FILE = str(Path("osm.net.xml").resolve())
ROUTE_FILE = str(Path("routes.rou.xml").resolve())
MODEL_FILE = "model_7.zip"
OUTPUT_XML_FILE = "predicted_tls.xml"

# Initialize the environment clearly
env = MultiTLEnvSingleAgent(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=False,
    num_seconds=1500,
    yellow_time=3,
    min_green=5,
    max_green=90,
    fixed_ts=True,
)

# Load the PPO model explicitly
model = PPO.load(MODEL_FILE, env=env)

# Reset environment and get current state
obs, _ = env.reset()

# Get predicted action from the model
predicted_action, _ = model.predict(obs, deterministic=True)

# Scale predicted PPO action back to durations explicitly
scaled_actions = env.min_duration + (predicted_action + 1.0) * (env.max_duration - env.min_duration) / 2.0

# Prepare XML file structure clearly
tls_element = ET.Element('additional')

idx = 0
for ts_id in env.ts_ids:
    logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
    phases = logic.phases

    # Assign predicted durations explicitly
    for phase_idx in env.phase_indices_per_tl[ts_id]:
        predicted_duration = float(np.clip(scaled_actions[idx], env.min_duration, env.max_duration))
        phases[phase_idx].duration = predicted_duration
        idx += 1

    # Construct XML structure clearly
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

# Save predicted durations clearly to XML file
tree = ET.ElementTree(tls_element)
ET.indent(tree, space="  ")
tree.write(OUTPUT_XML_FILE, encoding="utf-8", xml_declaration=True)

print(f"✅ Predicted TL phases successfully written to {OUTPUT_XML_FILE}")