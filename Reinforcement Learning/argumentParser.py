import argparse
class ArgumentParser:
    '''
    Argument parser class to parse command line arguments
    '''
    def __init__(self):
        '''
        Initialize the argument parser. It will accept the following arguments:
        --model_type: Type of the model. Default "PPO"
        --env_id: ID of the environment. Default "Humanoid-v5"
        --n_envs: Number of parallel environments. Default 1
        --seed: Random seed. Default 42
        --monitor_dir: Directory to save monitoring data. Default "./monitoring/"
        --no_record_video: Whether to record video. Default True
        --evaluate_model: Whether to evaluate the model. Default False
        --n_episode: Number of episodes to run. Default 10
        --learning_rate: Learning rate. Default 3e-4
        --n_steps: Number of steps to run for each environment. Default 2048
        --batch_size: Batch size. Default 64
        --n_epochs: Number of epochs. Default 10
        --gamma: Discount factor. Default 0.99
        --gae_lambda: GAE lambda. Default 0.95
        --clip_range: Clip range. Default 0.2
        --ent_coef: Entropy coefficient. Default 0.01
        --target_kl: Target KL divergence. Default 0.03
        --verbose: Verbosity level. Default 1
        --tensorboard_log: Directory to save tensorboard logs
        --save_path: Path to save the model. Default "./results/model/"
        --total_timesteps: Total timesteps to run the model. Default 500_000
        --device: Device to run the model. Default "cuda"
        --tunehp: Whether to perform hyperparameter search. Default False
        --train: Whether to train the model. Default True
        '''
        self.parser = argparse.ArgumentParser(description='Reinforcement Learning Environment Setup')
        self.parser.add_argument('--model_type', type=str, default='PPO', help='Type of the model. Model allowed: PPO, SAC, A2C')
        self.parser.add_argument('--env_id', type=str, default='Humanoid-v5', help='ID of the environment. ID allowed: Humanoid-v5, HalfCheetah-v5, Hopper-v5, Ant-v5')
        self.parser.add_argument('--save_path', type=str, default='./results/model', help='Path to save the model')
        self.parser.add_argument('--total_timesteps', type=int, default=500_000, help='Total timesteps to run the model')
        self.parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
        self.parser.add_argument('--tunehp', action='store_true', help='Whether to perform hyperparameter search')
        self.parser.add_argument('--notrain', action='store_false', help='Whether to not train the model')
        self.parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed')
        self.parser.add_argument('--env_monitor_dir', type=str, default='./monitoring/', help='Directory to save environment monitoring data')
        self.parser.add_argument('--no_record_video', action='store_false', help='Whether to record video. If not specified, video will be recorded')
        self.parser.add_argument('--evaluate_model', action='store_true', help='Whether to evaluate the model')
        '''
        self.parser.add_argument('--n_episodes', type=int, default=10, help='Number of episodes to run')
        self.parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
        self.parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps to run for each environment')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
        self.parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
        self.parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
        self.parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range')
        self.parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
        self.parser.add_argument('--target_kl', type=float, default=0.03, help='Target KL divergence')
        self.parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
        self.parser.add_argument('--tensorboard_log', type=str, default='./logs/', help='Directory to save tensorboard logs')
        '''
        

    def parse_args(self):
        return self.parser.parse_args()
