import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GreenPhaseLoggerCallback(BaseCallback):
    def __init__(self, log_every=1):
        super().__init__()
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True

        infos = self.locals.get("infos", [])

        for info in infos:
            if "ep_length_real" in info:
                self.logger.record("custom/ep_length_real", info["ep_length_real"])

            if "green_durations" in info:
                durations = np.array(info["green_durations"])
                normalized = (durations - 5) / (90 - 5)
                for i, dur in enumerate(normalized):
                    self.logger.record(f"phases/green_phase_{i}", float(dur))

            if "ep_rew_mean" in info:
                self.logger.record("custom/ep_rew_mean", info["ep_rew_mean"])

        self.logger.dump(self.num_timesteps)
        return True
