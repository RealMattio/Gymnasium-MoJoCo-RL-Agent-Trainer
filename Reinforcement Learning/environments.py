from stable_baselines3.common.env_util import make_vec_env

class EnvironmentInitializer:
    def __init__(self, env_id="Humanoid-v5", n_envs=1, seed=42, monitor_dir="./monitoring/"):
        """
        Initialize the environments.

        :param env_id: (str) The environment ID.
        :param n_envs: (int) The number of parallel environments to create.
        :param seed: (int) The seed for the random number generator.
        """
        self.env_id = env_id
        self.n_envs = n_envs
        self.seed = seed
        self.monitor_dir = monitor_dir

    def create_env(self):
        """
        Create and return the vectorized environment.

        :return: (VecEnv) The vectorized environment.
        """
        return make_vec_env(self.env_id, n_envs=self.n_envs, seed=self.seed, monitor_dir=self.monitor_dir)