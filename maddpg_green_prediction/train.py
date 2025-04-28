import os

import torch
import traci
import numpy as np
from maddpg_agent import MADDPGAgent
from torch.utils.tensorboard import SummaryWriter
import time

SUMO_BINARY = "sumo"  # Use GUI
SUMO_CFG = "../config/osm.sumocfg"

EPISODES = 1000
INTERVAL_DURATION = 60  # 10 minutes
MAX_ACTION = 90


def get_state(tl_id):
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    queue = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    wait = [traci.lane.getWaitingTime(lane) for lane in lanes]

    # Current active phase (one-hot)
    current_phase = traci.trafficlight.getPhase(tl_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
    phase_one_hot = [1 if i == current_phase else 0 for i in range(num_phases)]

    return np.array(queue + wait + phase_one_hot)


def compute_reward(tl_id):
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
    wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    return -(queue + 0.1 * wait)


def apply_phase_timings(tl_id, durations):
    logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    for i, phase in enumerate(logic.phases):
        phase.duration = durations[i]
    traci.trafficlight.setProgramLogic(tl_id, logic)


if __name__ == "__main__":
    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--no-step-log", "true",
                 "--no-warnings", "true", "--log", os.devnull])

    # tl_ids = traci.trafficlight.getIDList()
    tl_ids = ["632031937", "11644940310", "cluster10852700972_12154751993", "cluster_4110471701_632031939"]
    state_dims = []
    action_dims = []

    for tl_id in tl_ids:
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        n_lanes = len(lanes)
        n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)

        state_dim = n_lanes * 2 + n_phases
        action_dim = n_phases

        state_dims.append(state_dim)
        action_dims.append(action_dim)

    maddpg = MADDPGAgent(state_dims, action_dims, MAX_ACTION)
    writer = SummaryWriter(comment="-train-traffic-maddpg")

    traci.close()
    time.sleep(0.5)

    for ep in range(EPISODES):
        traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--no-step-log", "true",
                     "--no-warnings", "true", "--log", os.devnull])
        step = 0
        episode_rewards = np.zeros(len(tl_ids))

        while step < 600:
            states = [get_state(tl_id) for tl_id in tl_ids]
            actions = maddpg.select_action(states)

            actions = [np.clip(a, 5, MAX_ACTION) for a in actions]

            for tl_id, durations in zip(tl_ids, actions):
                apply_phase_timings(tl_id, durations)

            rewards = []
            next_states = []

            interval_end = step + INTERVAL_DURATION
            while step < interval_end:
                traci.simulationStep()
                step += 1

            for tl_id in tl_ids:
                rewards.append(compute_reward(tl_id))
                next_states.append(get_state(tl_id))

            dones = [step >= 3600] * len(tl_ids)

            maddpg.replay_buffer.add(states, actions, rewards, next_states, dones)
            maddpg.learn(batch_size=128)

            episode_rewards += np.array(rewards)

        traci.close()
        time.sleep(0.5)
        total_episode_reward = episode_rewards.sum()
        mean_episode_reward = episode_rewards.mean()

        writer.add_scalar("Episode/TotalReward", total_episode_reward, ep)
        writer.add_scalar("Episode/MeanRewardPerAgent", mean_episode_reward, ep)

        print(
            f"Episode {ep + 1} finished. Total reward: {total_episode_reward:.2f}, Mean reward: {mean_episode_reward:.2f}")


import json

SAVE_PATH = "./saved_models/"
os.makedirs(SAVE_PATH, exist_ok=True)

# Save actors
for idx, actor in enumerate(maddpg.actors):
    torch.save(actor.state_dict(), os.path.join(SAVE_PATH, f"actor_agent_{idx}.pth"))

# Save metadata
metadata = {"state_dims": state_dims, "action_dims": action_dims}
with open(os.path.join(SAVE_PATH, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print("✅ Models and metadata saved!")
