from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class GreenPhaseLoggerCallback(BaseCallback):
    def __init__(self, log_every=100):
        super().__init__()
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True  # ⏩ Skip logging this step

        # Unwrap to get the raw env
        env = self.training_env
        while hasattr(env, 'envs'):
            env = env.envs[0].env

        if hasattr(env, "last_durations") and env.last_durations is not None:
            normalized_durations = (np.array(env.last_durations) - 5) / (90 - 5)
            for i, dur in enumerate(normalized_durations):
                self.logger.record(f"phases/green_phase_{i}", float(dur))
            self.logger.dump(self.num_timesteps)

        return True
