from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import os

class EnvironmentInitializer:
    def __init__(self, model_type="PPO", env_id="Humanoid-v5", n_envs=1, seed=42, monitor_dir="./monitoring/"):
        """
        Initialize the environments.

        :param model_type: (str) The type of the model. Is required because different models require different environments.
            In particular if is chosed DQN it requires a discrete action space.
        :param env_id: (str) The environment ID.
        :param n_envs: (int) The number of parallel environments to create.
        :param seed: (int) The seed for the random number generator.
        """
        self.model_type = model_type
        self.env_id = env_id
        self.n_envs = n_envs
        self.seed = seed
        self.monitor_dir = monitor_dir

    def create_env(self):
        """
        Create and return the environment. If the model is DQN, it will create a discrete action space and will return a
        Monitor wrapper around the environment.

        :return: Environment.
        """
        
        if self.model_type == "DQN":
            env = gym.make(self.env_id)
            env = DiscreteActionWrapper(env)
            return Monitor(env)
        else:
            vec_env = make_vec_env(self.env_id, n_envs=self.n_envs, seed=self.seed, monitor_dir=self.monitor_dir)
            return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
    def create_video_env(self):
        """
        Create and return the environment for recording video.

        :return: Environment.
        """
        env = gym.make(self.env_id, render_mode="rgb_array")
        if self.model_type == "DQN":
            env = DiscreteActionWrapper(env)
        return env
    
    def save_norm_stats(env, save_path:str, name:str):
        """
        Save VecNormalize statistics for environments using either:
        - Direct VecNormalize wrapper
        - DummyVecEnv containing a VecNormalize wrapper
        - Discrete action-wrapped environments
        """        
        # Case 1: Environment is directly VecNormalized
        if isinstance(env, VecNormalize):
            env.save(os.path.join(save_path, name))
            print(f"VecNormalize statistics saved at {os.path.join(save_path, name)}")
            return
        
        # Case 2: Environment contains VecNormalize in its sub-wrappers
        if hasattr(env, 'venv') and isinstance(env.venv, VecNormalize):
            env.venv.save(os.path.join(save_path, name))
            print(f"VecNormalize statistics saved at {os.path.join(save_path, name)}")
            return
        
        # Case 3: Environment is DummyVecEnv with normalized observations
        if isinstance(env, DummyVecEnv):
            # Check if any wrapper in the chain is VecNormalize
            current_env = env
            while hasattr(current_env, 'venv'):
                current_env = current_env.venv
                if isinstance(current_env, VecNormalize):
                    current_env.save(os.path.join(save_path, name))
                    print(f"VecNormalize statistics saved at {os.path.join(save_path, name)}")
                    return
        
        print("VecNormalize statistics not found - skipping save")
    
    def load_norm_stats(load_path: str, env):
        """
        Load VecNormalize statistics into an existing environment
        """
        if os.path.exists(load_path):
            if isinstance(env, VecNormalize):
                env = VecNormalize.load(load_path, env)
            elif hasattr(env, 'venv'):
                current_env = env
                while hasattr(current_env, 'venv'):
                    if isinstance(current_env.venv, VecNormalize):
                        current_env.venv = VecNormalize.load(load_path, current_env.venv)
                        break
                    current_env = current_env.venv
            return env
        return env

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins_per_dim: int = 3):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        
        self.original_action_space = env.action_space
        low, high = env.action_space.low, env.action_space.high
        self.dim = env.action_space.shape[0]
        self.bins_per_dim = bins_per_dim
        
        # Create discrete action space
        self.n_actions = bins_per_dim ** self.dim
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        # Generate action mapping table
        self.action_table = np.array(np.meshgrid(
            *[np.linspace(low[i], high[i], bins_per_dim) for i in range(self.dim)]
        )).T.reshape(-1, self.dim)

    def action(self, action: int) -> np.ndarray:
        return self.action_table[action]
