from environments import EnvironmentInitializer
from argumentParser import ArgumentParser
from plotter import Plotter as pl
from model import Model, record_vec_video, record_dqn_video
import json, os


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
        model = Model(model_type=args.model_type, env=env, device=args.device, env_id=args.env_id)
        best_params = model.tune_hyperparameters()
        # Save the best parameters to a file
        print(f"Best parameters found: {best_params}")
        with open(f"./results/hyperparameters/{args.model_type}_{args.env_id}_best_params.json", "w") as file:
            json.dump(best_params, file)

    # If the user wants to train the model
    if args.no_train:
        # Create environment initializer
        env_initializer = EnvironmentInitializer(model_type=args.model_type, env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.env_monitor_dir)
        env = env_initializer.create_env()
        # Create the model
        model = Model(model_type=args.model_type, env=env, device=args.device, use_best_params=True, env_id=args.env_id)
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
            model = Model(model_type=args.model_type, env=env, device=args.device, use_best_params=True, env_id=args.env_id)
            model.load_model(f"{args.save_path}/{args.model_type}_{args.env_id}.zip")
        # Evaluate the model
        eval_env = EnvironmentInitializer.load_norm_stats(f"./vec_normalization/{args.model_type}_{args.env_id}.pkl", env)
        metrics = model.evaluate_model(eval_env, args.seed, n_eval_episodes=args.n_eval_episodes)
        model.save_metrics(metrics, f"{args.save_eval_path}")
    # If the user wants to record video
    if args.no_record_video:
        # Record video
        if args.model_type == "DQN":
            record_dqn_video(args.model_type, args.env_id, video_folder=args.video_folder)
        else:
            record_vec_video(args.model_type, args.env_id, video_folder=args.video_folder)
    
    if args.comparison_plot:
        if os.path.exists(f"{args.save_eval_path}/metrics_{args.env_id}.json"):
            with open(f"{args.save_eval_path}/metrics_{args.env_id}.json", "r") as file:
                metrics = json.load(file)
                all_algorithms = metrics.keys()
                # create a list of lists with the metrics for each algorithm
                metrics = [metrics[algo]["all_episodes_rewards"] for algo in all_algorithms]
                box_title = f"Comparison of Algorithms rewards during {args.n_eval_episodes} episodes for {args.env_id}"
                title = f"Rewards during episodes for {args.env_id}"
                pl.box_plot(metrics, all_algorithms, box_title, "Algorithms", "Rewards")
                pl.plot_rewards(metrics, all_algorithms, title, "Episodes", "Rewards")
        else:
            raise Exception(f"No metrics found for {args.env_id}. Plotting aborted.")

    




if __name__ == '__main__':
    main()
