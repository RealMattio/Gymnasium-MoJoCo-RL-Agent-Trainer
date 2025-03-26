from stable_baselines3 import DQN, PPO, SAC, A2C, TD3
#from sklearn.model_selection import ParameterGrid
import os, json, datetime
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
import optuna
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from environments import EnvironmentInitializer, DiscreteActionWrapper
import warnings
from typing import Dict

class Model:
    DEFAULT_PPO_PARAMS = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        }
    }
    DEFAULT_SAC_PARAMS = {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "policy_kwargs": dict(net_arch=[256, 256])
    }
    DEFAULT_A2C_PARAMS = {
        "learning_rate": 7e-4,
        "n_steps": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "policy_kwargs": dict(net_arch=[256, 256])
    }
    DEFAULT_DQN_PARAMS = {
                "learning_rate": 3e-4,
                "buffer_size": 1_000_000,
                "batch_size": 128,
                "gamma": 0.99,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.02,
                "policy_kwargs": dict(net_arch=[256, 256])
    }
    DEFAULT_TD3_PARAMS = {
                "learning_rate": 1e-3,
                "buffer_size": 1_000_000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "policy_kwargs": dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
            }
        
    
    def __init__(self, model_type='PPO', env=None, params:dict=None, device='cpu', use_best_params=False, env_id:str='Ant-v5'):
        '''
        Initialize the model with the given parameters
        :param model_type: Type of the model. Default "PPO"
        :param env: The environment to train the model on
        :param params: The parameters for the model. If specified the model will be initialized with these parameters
        :param device: The device to run the model on. Default "cpu", is better for model that don't require GPU
        :param use_best_params: Whether to use the best parameters found during hyperparameter tuning

        If params is not provided and use_best_param is False the model will be initialized with the default parameters
        If params is provided and use_best_param is True the model will be initialized with the provided parameters anyway
        '''
        self.model_type = model_type
        self.device = device
        self.env = env
        self.env_id = env_id
        if self.env is None:
            warnings.warn("\nEnvironment not provided. Make sure you are just recording the video otherwise the program will crash\n", UserWarning)
        if params is not None:
            self.best_params = params
        # if the best_params.json file exist, store the best parameters from that file
        elif use_best_params and os.path.exists(f"./results/hyperparameters/{self.model_type}_{self.env_id}_best_params.json"):
            with open(f"./results/hyperparameters/{self.model_type}_{self.env_id}_best_params.json", "r") as file:
                self.best_params = json.load(file)
            print(f"Best parameters found! Model initialized with these parameters.")
        else:
            print("No best parameters found")
            self.best_params = None
        self.model = self._initialize_model()

    def _initialize_model(self, params=None):
        if params is None:
            if self.model_type == 'A2C':
                if self.best_params is None:
                    return A2C('MlpPolicy', self.env, verbose=0, device=self.device, **self.DEFAULT_A2C_PARAMS)
                else:
                    return A2C('MlpPolicy', self.env, verbose=0, device=self.device, **self.best_params)
            elif self.model_type == 'PPO':
                if self.best_params is None:
                    return PPO('MlpPolicy', self.env, verbose=0, device=self.device, **self.DEFAULT_PPO_PARAMS)
                else:
                    return PPO('MlpPolicy', self.env, verbose=0, device=self.device, **self.best_params)
            elif self.model_type == 'SAC':
                if self.best_params is None:
                    return SAC('MlpPolicy', self.env, verbose=0, device=self.device, **self.DEFAULT_SAC_PARAMS)
                else:
                    return SAC('MlpPolicy', self.env, verbose=0, device=self.device, **self.best_params) 
            elif self.model_type == 'DQN':
                if self.best_params is None:
                    return DQN('MlpPolicy', self.env, verbose=0, device=self.device, **self.DEFAULT_DQN_PARAMS)
                else:
                    return DQN('MlpPolicy', self.env, verbose=0, device=self.device, **self.best_params)
            elif self.model_type == 'TD3':
                if self.best_params is None:
                    return TD3('MlpPolicy', self.env, verbose=0, device=self.device, **self.DEFAULT_TD3_PARAMS)
                else:
                    return TD3('MlpPolicy', self.env, verbose=0, device=self.device, **self.best_params)
            elif self.model_type == 'random':
                warnings.warn("Random model selected. No model will be initialized. Make sure you are performing just evaluation or recording video. Otherwise the program will crash")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        else:
            if self.model_type == 'A2C':
                return A2C('MlpPolicy', self.env, verbose=0, device=self.device, **params)
            elif self.model_type == 'PPO':
                return PPO('MlpPolicy', self.env, verbose=0, device=self.device, **params)
            elif self.model_type == 'SAC':
                return SAC('MlpPolicy', self.env, verbose=0, device=self.device, **params)
            elif self.model_type == 'DQN':
                return DQN('MlpPolicy', self.env, verbose=0, device=self.device, **params)
            elif self.model_type == 'TD3':
                return TD3('MlpPolicy', self.env, verbose=0, device=self.device, **params)
            elif self.model_type == 'random':
                warnings.warn("Random model selected. No model will be initialized. Make sure you are performing just evaluation or recording video. Otherwise the program will crash")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    def load_model(self, path, record=False, env=None):
        if self.model_type == 'A2C':
            if record and env is not None:
                self.model = A2C.load(path, env=env)
            elif record and env is None:
                raise Exception("Environment not provided")
            else:
                self.model = A2C.load(path)
        elif self.model_type == 'PPO':
            if record and env is not None:
                self.model = PPO.load(path, env=env)
            elif record and env is None:
                raise Exception("Environment not provided")
            else:
                self.model = PPO.load(path)
        elif self.model_type == 'SAC':
            if record and env is not None:
                self.model = SAC.load(path, env=env)
            elif record and env is None:
                raise Exception("Environment not provided")
            else:
                self.model = SAC.load(path)
        elif self.model_type == 'DQN':
            if record and env is not None:
                self.model = DQN.load(path, env=env)
            elif record and env is None:
                raise Exception("Environment not provided")
            else:
                self.model = DQN.load(path)
        elif self.model_type == 'TD3':
            if record and env is not None:
                self.model = TD3.load(path, env=env)
            elif record and env is None:
                raise Exception("Environment not provided")
            else:
                self.model = TD3.load(path)
        elif self.model_type == 'random':
            warnings.warn("Random model selected. No model will be loaded. Make sure you are performing just evaluation or recording video. Otherwise the program will crash", UserWarning)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return self.model

    def tune_hyperparameters(self, n_tuning_trials=20) -> dict:
        policy_arch = {
                "small": [64, 64],
                "medium": [256, 256],
                "large": [512, 512]
        }
        def objective(trial: optuna.Trial) -> float:
            policy_arch = {
                "small": [64, 64],
                "medium": [256, 256],
                "large": [512, 512],
                "custom": [
                    trial.suggest_categorical(f"layer_{i}", [64, 128, 256, 512])
                    for i in range(trial.suggest_int("num_layers", 1, 4))
                ]
            }
            arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large", "custom"])
            hyperparams = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            }
            # Knowing the model type, we can add more hyperparameters
            if self.model_type == "DQN":
                hyperparams = {
                    "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 1_000_000]),
                    "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                    "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
                    "exploration_initial_eps": trial.suggest_float("exploration_initial_eps", 0.9, 1.0),
                    "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
                    "policy_kwargs": {"net_arch": policy_arch[arch_choice]}
                }
            elif self.model_type == "PPO":
                hyperparams = {
                    "n_steps": trial.suggest_int("n_steps", 512, 4096, step=512),
                    "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
                    "n_epochs": trial.suggest_int("n_epochs", 3, 30),
                    "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
                    "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                    "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                    "policy_kwargs": {"net_arch": policy_arch[arch_choice]}
                }
            elif self.model_type == "A2C":
                hyperparams = {
                    "n_steps": trial.suggest_int("n_steps", 5, 500),
                    "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
                    "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                    "policy_kwargs": {"net_arch": policy_arch[arch_choice]}
                }
            elif self.model_type == "SAC":
                hyperparams = {
                    "buffer_size": trial.suggest_int("buffer_size", 500_000, 3_000_000, step=500_000),
                    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
                    "tau": trial.suggest_float("tau", 0.001, 0.02),
                    "policy_kwargs": {
                        "net_arch": {
                            "pi": policy_arch[arch_choice],
                            "qf": policy_arch[arch_choice]
                        }
                    }
                }
            elif self.model_type == "TD3":
                hyperparams = {
                    "buffer_size": trial.suggest_int("buffer_size", 500_000, 3_000_000, step=500_000),
                    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
                    "tau": trial.suggest_float("tau", 0.001, 0.02),
                    "policy_delay": trial.suggest_int("policy_delay", 1, 4),
                    "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.3),
                    "target_noise_clip": trial.suggest_float("target_noise_clip", 0.1, 0.3),
                    "policy_kwargs": {
                        "net_arch": {
                            "pi": policy_arch[arch_choice],
                            "qf": policy_arch[arch_choice]
                        }
                    }
                }

            model = self._initialize_model(params=hyperparams)
            eval_callback = EvalCallback(
                self.env,
                best_model_save_path=None,
                eval_freq=5000,
                deterministic=True
            )
            model.learn(total_timesteps=100000, callback=eval_callback)
            return np.mean(eval_callback.last_mean_reward)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_tuning_trials)
        # rewrite net_arch param to be a list of integers
        best_params = study.best_params.copy()
        if "net_arch" in best_params and best_params["net_arch"] == "custom":
            net_arch = []
            for i in range(best_params["num_layers"]):
                net_arch.append(best_params[f"layer_{i}"])
                best_params.pop(f"layer_{i}")
            best_params["policy_kwargs"] = {"net_arch": net_arch}
            best_params.pop("net_arch")
            best_params.pop("num_layers")
        else:
            best_params["policy_kwargs"] = {"net_arch":policy_arch[best_params["net_arch"]]}
            best_params.pop("net_arch")
            for i in range(best_params["num_layers"]):
                best_params.pop(f"layer_{i}")
            best_params.pop("num_layers")
        return best_params

    def evaluate_model(self, eval_env, seed, n_eval_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluates a RL model's performance with proper VecEnv handling.
        """
        if self.model_type == "random":
            return self.run_random_policy(seed, n_episodes=n_eval_episodes)
        # Ensure environment is vectorized
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        
        episode_rewards = []
        current_rewards = np.zeros(eval_env.num_envs)
        episode_counts = np.zeros(eval_env.num_envs, dtype=int)

        obs = eval_env.reset()
        
        while len(episode_rewards) < n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(actions)
            
            current_rewards += rewards
            
            for env_idx in range(eval_env.num_envs):
                if dones[env_idx]:
                    episode_rewards.append(current_rewards[env_idx])
                    episode_counts[env_idx] += 1
                    current_rewards[env_idx] = 0
                    obs = eval_env.reset()

        rewards_array = np.array(episode_rewards)
        metrics = {
            "mean_episode_reward": float(np.mean(rewards_array)),
            "std_episode_reward": float(np.std(rewards_array)),
            "variance_episode_reward": float(np.var(rewards_array)),
            "total_episodes_reward": float(np.sum(rewards_array)),
            "max_episode_reward": float(np.max(rewards_array)),
            "min_episode_reward": float(np.min(rewards_array)),
            "episode_count": len(episode_rewards),
            "all_episodes_rewards": [float(rew) for rew in episode_rewards],
            "episodes_per_env": episode_counts.tolist()
        }
        print(f"\nEvaluation of model {self.model_type} on environment {self.env_id} completed.")
        return metrics

    def save_metrics(self, metrics:dict, path:str="./results/evaluation"):
        if os.path.exists(f"{path}/metrics_{self.env_id}.json"):
            with open(f"{path}/metrics_{self.env_id}.json", "r") as file:
                data = json.load(file)
                data[f"{self.model_type}"] = metrics
            with open(f"{path}/metrics_{self.env_id}.json", "w") as file:
                json.dump(data, file)
        else:
            os.makedirs(path, exist_ok=True)
            data = {}
            with open(f"{path}/metrics_{self.env_id}.json", "w") as file:
                data[f"{self.model_type}"] = metrics
                json.dump(data, file)
        print(f"Metrics saved at {path}/metrics_{self.env_id}.json")
    
    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved at {path}")
    
    def run_random_policy(self, seed, n_episodes=10):
        env = gym.make(self.env_id)
        env = Monitor(env)
        episodes_rewards = []
        for episode in range(n_episodes):
            obs= env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            current_reward = 0
            while not terminated and not truncated:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                current_reward += reward
            episodes_rewards.append(current_reward)
        metrics = {
            "mean_episode_reward": float(np.mean(episodes_rewards)),
            "std_episode_reward": float(np.std(episodes_rewards)),
            "variance_episode_reward": float(np.var(episodes_rewards)),
            "total_episodes_reward": float(np.sum(episodes_rewards)),
            "max_episode_reward": float(np.max(episodes_rewards)),
            "min_episode_reward": float(np.min(episodes_rewards)),
            "episode_count": len(episodes_rewards),
            "all_episodes_rewards": [float(rew) for rew in episodes_rewards]
        }
        return metrics

@staticmethod          
def record_vec_video(model_type:str, env_id:str, video_folder:str="./results/videos/", video_length:int=500, seed=45):
    # Create a new environment for recording
    test_env = make_vec_env(env_id=lambda: gym.make(env_id, render_mode="rgb_array"), n_envs=1, seed=seed)

    # Load VecNormalize statistics for the test environment
    if os.path.exists(f"./vec_normalization/{model_type}_{env_id}.pkl"):
        test_env = VecNormalize.load(f"./vec_normalization/{model_type}_{env_id}.pkl", test_env)
    
    prefix = f"{model_type}_{env_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Record video of the model's performance
    test_env = VecVideoRecorder(test_env, video_folder, record_video_trigger=lambda x: x == 0, video_length=video_length, name_prefix=prefix)
    model = Model(model_type=model_type, env=test_env, use_best_params=True)
    model = model.load_model(f"./results/model/{model_type}_{env_id}.zip", record=True, env=test_env)

    # Reset the environment
    obs = test_env.reset()

    # Run the model in the environment and record the video
    for _ in tqdm(range(video_length)):
        if model_type == "random":
            action = [test_env.action_space.sample()]
        else:
            action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        # exit recording if episode is terminated or truncated
        if dones:
            break

    # Close the environment
    test_env.close()
    print(f"Video recorded successfully. Saved at {video_folder}/{prefix}-step-0-to-step-500.mp4")

@staticmethod
def record_dqn_video(model_type:str, env_id:str, video_folder:str="./results/videos/", video_length:int=500, seed=45):
    # Create a new environment for recording
    model_type = "DQN"
    env = gym.make(env_id, render_mode="rgb_array")
    env = DiscreteActionWrapper(env)
    env = Monitor(env)
    model = Model(model_type=model_type, env=env, use_best_params=True)
    model = model.load_model(f"./results/model/{model_type}_{env_id}.zip", record=True, env=env)
    prefix = f"{model_type}_{env_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix=prefix,
        disable_logger=True
    )
    # Initialize tracking
    frames = []
    try:
        obs, info = env.reset(seed=seed)
        for _ in tqdm(range(video_length)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break
            # Capture frame directly from render
            frame = env.render()
            if frame is not None:
                frames.append(frame)    
    finally:
        env.close()
        print(f"Video recorded successfully. Saved at {video_folder}/{prefix}-episode-0.mp4")
