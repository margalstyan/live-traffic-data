from stable_baselines3.common.callbacks import BaseCallback

class GreenPhaseLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        if hasattr(env, "last_durations") and env.last_durations is not None:
            for i, dur in enumerate(env.last_durations):
                self.logger.record(f"phases/green_phase_{i}", dur)
        return True
