import os
import gymnasium as gym
import numpy as np
import multiprocessing as mp
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from env import SUMOMultiAgentEnv

# === Config ===
SUMO_CFG = "osm.sumocfg"
ROUTE_CSV = "routes.csv"
TOTAL_EPISODES = 20000
EPISODES_PER_WORKER = 1
NUM_WORKERS = 8
BATCH_SIZE = 100
SAVE_EVERY = 100
LEARNING_RATE = 3e-4
BASE_LOG_DIR = "./logs_marl_parallel"
BASE_MODEL_DIR = "./sac_agents_parallel"

os.makedirs(BASE_LOG_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

class DummySingleAgentEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

def run_worker_with_return(args):
    worker_id, episodes, model_dir = args
    port = 8813 + worker_id
    env = SUMOMultiAgentEnv(SUMO_CFG, ROUTE_CSV, sumo_port=port)
    env.tripinfo_path = f"xml/tripinfo_{worker_id}.xml"
    env.generated_route_path = f"xml/routes_generated_{worker_id}.rou.xml"
    tls_ids = env.tls_ids

    agents = {}
    for tls_id in tls_ids:
        model_path = os.path.join(model_dir, tls_id, "latest.zip")
        agents[tls_id] = SAC.load(
            model_path,
            env=None,
            custom_objects={"lr_schedule": lambda _: LEARNING_RATE},
        )

    transitions = []
    for _ in range(episodes):
        obs, _ = env.reset()
        actions = {tls_id: agents[tls_id].predict(obs[tls_id], deterministic=True)[0] for tls_id in tls_ids}
        next_obs, rewards, dones, _, infos = env.step(actions)

        for tls_id in tls_ids:
            transitions.append((tls_id, obs[tls_id], actions[tls_id], rewards[tls_id],
                                next_obs[tls_id], dones[tls_id], infos[tls_id][0] if infos[tls_id] else {}))

    print(f"[Worker {worker_id}] Finished {episodes} episodes")
    return transitions

def parallel_train_with_pool():
    env = SUMOMultiAgentEnv(SUMO_CFG, ROUTE_CSV)
    tls_ids = env.tls_ids

    agents = {}
    loggers = {}
    episode_counters = {tls_id: 0 for tls_id in tls_ids}

    for tls_id in tls_ids:
        obs_space = env.agent_observation_spaces[tls_id]
        act_space = env.agent_action_spaces[tls_id]
        dummy_env = DummySingleAgentEnv(obs_space, act_space)

        log_path = os.path.join(BASE_LOG_DIR, tls_id)
        os.makedirs(log_path, exist_ok=True)
        logger = configure(log_path, ["stdout", "tensorboard"])

        model = SAC(
            policy="MlpPolicy",
            env=dummy_env,
            learning_rate=LEARNING_RATE,
            buffer_size=50000,
            batch_size=BATCH_SIZE,
            verbose=0,
            tensorboard_log=log_path
        )
        model.set_logger(logger)
        agents[tls_id] = model
        loggers[tls_id] = logger

        os.makedirs(os.path.join(BASE_MODEL_DIR, tls_id), exist_ok=True)
        model.save(os.path.join(BASE_MODEL_DIR, tls_id, "latest.zip"))

    episodes_done = 0
    cycle = 0

    while episodes_done < TOTAL_EPISODES:
        print(f"\nCycle {cycle} - Launching {NUM_WORKERS} workers using Pool")
        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.map(run_worker_with_return, [(wid, EPISODES_PER_WORKER, BASE_MODEL_DIR) for wid in range(NUM_WORKERS)])

        samples_per_tls = {tls_id: [] for tls_id in tls_ids}
        infos_per_tls = {tls_id: [] for tls_id in tls_ids}

        for worker_transitions in results:
            for tls_id, obs, action, reward, next_obs, done, info in worker_transitions:
                samples_per_tls[tls_id].append((obs, action, reward, next_obs, done))
                infos_per_tls[tls_id].append(info)

        for tls_id in tls_ids:
            agent = agents[tls_id]
            logger = loggers[tls_id]
            current_step = episode_counters[tls_id]

            for i, (obs, action, reward, next_obs, done) in enumerate(samples_per_tls[tls_id]):
                agent.replay_buffer.add(obs, next_obs, action, reward, done, infos=[{}])
                logger.record("train/reward", reward)

            if len(samples_per_tls[tls_id]) >= BATCH_SIZE:
                # Train and dump internal metrics (actor/critic/entropy losses) automatically
                agent.train(gradient_steps=len(samples_per_tls[tls_id]), batch_size=BATCH_SIZE)

                if agent.logger is not None:
                    # Retrieve SAC loss info
                    logs = agent.logger.name_to_value
                    if "train/critic_loss" in logs:
                        logger.record("train/critic_loss", logs["train/critic_loss"])
                    if "train/actor_loss" in logs:
                        logger.record("train/actor_loss", logs["train/actor_loss"])
                    if "train/entropy" in logs:
                        logger.record("train/entropy", logs["train/entropy"])
                    logger.dump(step=episode_counters[tls_id])

            for i, info in enumerate(infos_per_tls[tls_id]):
                logger.record("env/avg_wait", info.get("avg_wait", 0))
                logger.record("env/avg_trip_duration", info.get("avg_duration", 0))
                logger.record("env/vehicle_count", info.get("vehicle_count", 0))
                logger.dump(step=current_step + i)

            episode_counters[tls_id] += len(infos_per_tls[tls_id])

        episodes_done += NUM_WORKERS * EPISODES_PER_WORKER
        cycle += 1

        if episodes_done % SAVE_EVERY == 0:
            print(f"Saving models at episode {episodes_done}")
            for tls_id in tls_ids:
                agents[tls_id].save(os.path.join(BASE_MODEL_DIR, tls_id, "latest.zip"))
                agents[tls_id].save(os.path.join(BASE_MODEL_DIR, tls_id, f"ep_{episodes_done}.zip"))

if __name__ == "__main__":
    parallel_train_with_pool()
