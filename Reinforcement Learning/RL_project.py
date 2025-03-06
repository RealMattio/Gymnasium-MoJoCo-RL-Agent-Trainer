# Modified video recording function with render_mode fix
def record_video(model, env_id, video_dir, timestamp):
    # Create environment with rgb_array render mode
    base_env = gym.make(env_id, render_mode="rgb_array")  # <-- FIX HERE
    
    # Wrap in RecordVideo with timestamp
    video_env = RecordVideo(
        base_env,
        video_folder=video_dir,
        name_prefix=f"humanoid_{timestamp}",
        episode_trigger=lambda x: True
    )
    
    # Apply normalization
    video_vec_env = DummyVecEnv([lambda: video_env])
    video_vec_env = VecNormalize.load(f"{log_dir}/vecnormalize.pkl", video_vec_env)
    
    # Run episode
    obs = video_vec_env.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs)
        obs, _, done, _ = video_vec_env.step(action)
    
    video_vec_env.close()
    return f"humanoid_{timestamp}-episode-0.mp4"

# Full corrected code with video recording
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import os
import datetime

# Set seed and create directories
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

log_dir = "./humanoid_ppo/"
video_dir = "./videos/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Create and normalize environments
env_id = "Humanoid-v4"
num_envs = 8

vec_env = make_vec_env(env_id, n_envs=num_envs)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

# Initialize and train model
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    target_kl=0.03,
    verbose=1,
    tensorboard_log=log_dir
)

model.learn(total_timesteps=100_000, callback=EvalCallback(vec_env, best_model_save_path=log_dir, eval_freq=5000))

# Save model
model.save(f"{log_dir}/ppo_humanoid_final")
vec_env.save(f"{log_dir}/vecnormalize.pkl")

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Record video with fixed render mode
del model
model = PPO.load(f"{log_dir}/ppo_humanoid_final")

def record_video(model, env_id, video_dir, timestamp):
    base_env = gym.make(env_id, render_mode="rgb_array")  # Explicit render mode
    video_env = RecordVideo(
        base_env,
        video_folder=video_dir,
        name_prefix=f"humanoid_{timestamp}",
        episode_trigger=lambda x: True
    )
    
    video_vec_env = DummyVecEnv([lambda: video_env])
    video_vec_env = VecNormalize.load(f"{log_dir}/vecnormalize.pkl", video_vec_env)
    
    obs = video_vec_env.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs)
        obs, _, done, _ = video_vec_env.step(action)
    
    video_vec_env.close()
    return f"humanoid_{timestamp}-episode-0.mp4"

video_filename = record_video(model, env_id, video_dir, timestamp)
print(f"Successfully recorded video: {video_filename}")