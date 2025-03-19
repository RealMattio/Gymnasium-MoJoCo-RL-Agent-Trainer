from stable_baselines3 import DQN, PPO, SAC, A2C
#from sklearn.model_selection import ParameterGrid
import os, json, datetime
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
import optuna
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from environments import EnvironmentInitializer
import warnings

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
    def __init__(self, model_type='PPO', env=None, params:dict=None, device='cpu', use_best_params=False):
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
        if self.env is None:
            warnings.warn("Environment not provided. Make sure you are just recording the video otherwise the program will crash")
        if params is not None:
            self.best_params = params
        # if the best_params.json file exist, store the best parameters from that file
        elif use_best_params and os.path.exists(f"./results/hyperparameters/{self.model_type}_best_params.json"):
            with open(f"./results/hyperparameters/{self.model_type}_best_params.json", "r") as file:
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
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    def load_model(self, path):
        if self.model_type == 'A2C':
            self.model = A2C.load(path)
        elif self.model_type == 'PPO':
            self.model = PPO.load(path)
        elif self.model_type == 'SAC':
            self.model = SAC.load(path)
        elif self.model_type == 'DQN':
            self.model = DQN.load(path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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

    def evaluate_model(self, model):
        # Implement your evaluation logic here
        # For example, return the mean reward over several episodes
        return 0
    
    def record_video(self, env_id, video_dir="./results/videos", episodes=5, seed=42, max_steps=10000):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name_prefix = f"{self.model_type}_{env_id}_{timestamp}"
        base_env = gym.make(env_id, render_mode="rgb_array")  # Explicit render mode
        base_env = EnvironmentInitializer(env_id=env_id, n_envs=1, seed=seed, model_type=self.model_type).create_video_env()
        video_env = RecordVideo(
            base_env,
            video_folder=video_dir,
            name_prefix=name_prefix,
            episode_trigger=lambda x: True
        )
        video_env = RecordEpisodeStatistics(video_env)  # Track episode statistics (e.g., rewards, length)
        video_env = TimeLimit(video_env, max_episode_steps=max_steps) 
        video_env.reset(seed=seed)
        
        #video_vec_env = DummyVecEnv([lambda: video_env])
        #video_vec_env = VecNormalize.load(f"vec_normalization/vecnormalize.pkl", video_vec_env)
        
        rewards = {}
        print(f"Recording videos for {episodes} episodes")
        for episode in tqdm(range(episodes), desc="Episodes"):
            obs, _ = video_env.reset(seed=seed+episode)
            #obs = video_vec_env.reset()
            done = False
            episode_reward = 0
            n_action = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = video_env.step(action)
                #obs, _, terminated, truncated = video_vec_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                n_action += 1
            rewards[f"reward_episode_{episode}"] = episode_reward
            rewards[f"n_action_episode_{episode}"] = n_action
        #video_vec_env.close()
        video_env.close()
        print(f"Episods completed. Reward obtained: {rewards}")
        print(f"Saved videos \"{name_prefix}-episode-xx.mp4\" in \"{video_dir}\"")
    
    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved at {path}")
        