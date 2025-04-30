from stable_baselines3.common.callbacks import BaseCallback

class GreenPhaseLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            env = self.training_env.envs[0]
            if hasattr(env, "get_current_green_durations"):
                durations = env.get_current_green_durations()
                for i, dur in enumerate(durations):
                    self.logger.record(f"phases/green_phase_{i}", dur)
        return True
