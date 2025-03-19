from environments import EnvironmentInitializer
from argumentParser import ArgumentParser
from model import Model
import json


def main():
    '''
    Main function to run the program:
    This function will create the pipeline to train the agent
    '''
    # Parse command line arguments
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    env = None
    model = None

    # If the user wants to perform hyperparameter search
    if args.tunehp:
        # Perform hyperparameter search
        env_initializer = EnvironmentInitializer(model_type=args.model_type, env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.env_monitor_dir)
        env = env_initializer.create_env()
        model = Model(model_type=args.model_type, env=env, device=args.device)
        best_params = model.tune_hyperparameters()
        # Save the best parameters to a file
        print(f"Best parameters found: {best_params}")
        with open(f"./results/hyperparameters/{args.model_type}_best_params.json", "w") as file:
            json.dump(best_params, file)

    # If the user wants to train the model
    if args.notrain:
        # Create environment initializer
        env_initializer = EnvironmentInitializer(model_type=args.model_type, env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.env_monitor_dir)
        env = env_initializer.create_env()
        # Create the model
        model = Model(model_type=args.model_type, env=env, device=args.device, use_best_params=True)
        # Train the model
        model.learn(total_timesteps=args.total_timesteps)
        # Save the model and environment
        model_dir = args.save_path
        model.save_model(f"{model_dir}/{args.model_type}_{args.env_id}")
        EnvironmentInitializer.save_norm_stats(env, f"./vec_normalization", f"{args.model_type}_{args.env_id}.pkl")
       
    # If the user wants to evaluate the model
    if args.evaluate_model:
        if model is None and env is None:
            env_initializer = EnvironmentInitializer(model_type=args.model_type, env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.env_monitor_dir)
            env = env_initializer.create_env()
            # load the normalized statistics
            model = Model(model_type=args.model_type, env=env, device=args.device, use_best_params=True)
            model.load_model(f"{args.save_path}/{args.model_type}_{args.env_id}.zip")
        # Evaluate the model
        env = EnvironmentInitializer.load_norm_stats(f"./vec_normalization/{args.model_type}_{args.env_id}.pkl", env)
        model.evaluate_model(n_episodes=args.n_episodes)
    
    # If the user wants to record video
    if args.no_record_video:
        if model is None and env is None:
            env_initializer = EnvironmentInitializer(model_type=args.model_type, env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.env_monitor_dir)
            env = env_initializer.create_env()
            model = Model(model_type=args.model_type, env=env, device=args.device, use_best_params=True)
            model.load_model(f"{args.save_path}/{args.model_type}_{args.env_id}.zip")
        # Record video
        model.record_video(args.env_id)
    




if __name__ == '__main__':
    main()
