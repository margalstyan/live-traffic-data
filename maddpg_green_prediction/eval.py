import json

import torch
import traci
import numpy as np
import os
from maddpg_agent import Actor

SUMO_BINARY = "sumo-gui"
SUMO_CFG = "../config/osm.sumocfg"
MAX_ACTION = 90

# Get dimensions again from your network
traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--no-step-log", "true"])
tl_ids = ["632031937", "11644940310", "cluster10852700972_12154751993", "cluster_4110471701_632031939"]
state_dims, action_dims = [], []

for tl_id in tl_ids:
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
    state_dims.append(len(lanes)*2 + num_phases)
    action_dims.append(num_phases)
traci.close()

# Load metadata
with open("./saved_models/metadata.json", "r") as f:
    metadata = json.load(f)

state_dims = metadata["state_dims"]
action_dims = metadata["action_dims"]

# Load actors with correct dimensions
actors = []
for idx in range(len(state_dims)):
    actor = Actor(state_dims[idx], action_dims[idx], MAX_ACTION)
    actor.load_state_dict(torch.load(f"./saved_models/actor_agent_{idx}.pth"))
    actor.eval()
    actors.append(actor)

# Define get_state as before
def get_state(tl_id):
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    queue = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    wait = [traci.lane.getWaitingTime(lane) for lane in lanes]

    current_phase = traci.trafficlight.getPhase(tl_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
    phase_one_hot = [1 if i == current_phase else 0 for i in range(num_phases)]

    return np.array(queue + wait + phase_one_hot)

# Predict and apply once (example for next 10-minutes static plan)
traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--no-step-log", "true"])

states = [get_state(tl_id) for tl_id in tl_ids]

with torch.no_grad():
    actions = [actors[i](torch.FloatTensor(states[i])).numpy() for i in range(len(tl_ids))]
    actions = [np.clip(a, 5, MAX_ACTION) for a in actions]

for tl_id, durations in zip(tl_ids, actions):
    logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    for phase_idx, phase in enumerate(logic.phases):
        phase.duration = durations[phase_idx]
    traci.trafficlight.setProgramLogic(tl_id, logic)

# Run the simulation for the next 10 minutes
step, INTERVAL_DURATION = 0, 600
while step < INTERVAL_DURATION:
    traci.simulationStep()
    step += 1

traci.close()
print("✅ Evaluation finished!")
