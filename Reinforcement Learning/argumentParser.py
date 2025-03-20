import argparse
class ArgumentParser:
    '''
    Argument parser class to parse command line arguments
    '''
    def __init__(self):
        '''
        Initialize the argument parser. It will accept the following arguments:
        --model_type: Type of the model. Default "PPO"
        --env_id: ID of the environment. Default "Ant-v5"
        --save_path: Path to save the model. Default "./results/model/"
        --save_eval_path: Path to save the evaluation metrics. Default "./results/evaluation"
        --total_timesteps: Total timesteps to run the model. Default 500_000
        --device: Device to run the model. Default "cuda"
        --tunehp: Whether to perform hyperparameter search. Default False
        --no_train: Whether to train the model. Default True
        --n_envs: Number of parallel environments. Default 1
        --seed: Random seed. Default 42
        --env_monitor_dir: Directory to save environment monitoring data. Default "./monitoring/"
        --no_record_video: Whether to record video. Default True
        --evaluate_model: Whether to evaluate the model. Default False
        --n_eval_episode: Number of evaluation episodes to run. Default 10
        --comparison_plot: Whether to plot the comparison. Default False
        '''
        self.parser = argparse.ArgumentParser(description='Reinforcement Learning Environment Setup')
        self.parser.add_argument('--model_type', type=str, default='PPO', help='Type of the model. Model allowed: PPO, SAC, A2C, DQN, random. If random, will be performed a random policy')
        self.parser.add_argument('--env_id', type=str, default='Ant-v5', help='ID of the environment. ID allowed: Humanoid-v5, HalfCheetah-v5, Hopper-v5, Ant-v5')
        self.parser.add_argument('--save_path', type=str, default='./results/model', help='Path to save the model')
        self.parser.add_argument('--save_eval_path', type=str, default='./results/evaluation', help='Path to save the evaluation metrics. Default "./results/evaluation"')
        self.parser.add_argument('--total_timesteps', type=int, default=500_000, help='Total timesteps to run the model')
        self.parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
        self.parser.add_argument('--tunehp', action='store_true', help='Whether to perform hyperparameter search')
        self.parser.add_argument('--no_train', action='store_false', help='Whether to not train the model')
        self.parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed')
        self.parser.add_argument('--env_monitor_dir', type=str, default='./monitoring/', help='Directory to save environment monitoring data')
        self.parser.add_argument('--no_record_video', action='store_false', help='Whether to record video. If not specified, video will be recorded')
        self.parser.add_argument('--evaluate_model', action='store_true', help='Whether to evaluate the model')
        self.parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of evaluation episodes to run')
        self.parser.add_argument('--comparison_plot', action='store_true', help='Whether to plot the comparison')
        

    def parse_args(self):
        return self.parser.parse_args()
