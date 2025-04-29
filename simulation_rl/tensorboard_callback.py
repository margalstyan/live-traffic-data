import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in TensorBoard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        super()._on_step()
        # Log mean reward
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer if "r" in ep_info])
            self.logger.record('train/mean_reward', mean_reward)

        # Additional logging if env exposes info
        if self.locals.get('infos'):
            infos = self.locals['infos']
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0]
                if 'mean_duration_error' in info:
                    self.logger.record('train/mean_duration_error', info['mean_duration_error'])
                    if self.verbose > 0:
                        print(f"Mean Duration Error: {info['mean_duration_error']:.2f} seconds")
        return True


from stable_baselines3.common.callbacks import BaseCallback

class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when at least 80% of the routes have simulation durations within 20% relative error
    (error based on max(simulated, target)).
    """

    def __init__(self, required_success_rate=0.8, tolerance=0.2, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.required_success_rate = required_success_rate
        self.tolerance = tolerance

    def _on_step(self) -> bool:
        if self.locals.get('infos'):
            infos = self.locals['infos']
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0]
                if 'mean_durations' in info:
                    mean_durations = info['mean_durations']
                    successes = 0
                    total = 0
                    for route_id, sim_duration in mean_durations.items():
                        expected = self.training_env.envs[0].unwrapped.routes[route_id]['target_duration']
                        if expected > 0 and sim_duration > 0:
                            relative_error = abs(sim_duration - expected) / max(sim_duration, expected)
                            print(
                                f"Expected: {expected:.2f}, Simulated: {sim_duration:.2f}, Error: {relative_error:.2f}")
                            if relative_error <= self.tolerance:
                                successes += 1
                            total += 1
                    if total > 0:
                        success_rate = successes / total
                        self.logger.record('train/success_rate', success_rate)
                        if self.verbose > 0:
                            print(f"✅ Success Rate: {success_rate:.2f}")
                        if success_rate >= self.required_success_rate:
                            print(f"🎯 Success rate {success_rate:.2f} >= {self.required_success_rate:.2f}. Stopping training!")
                            return False  # Stops training
        return True
