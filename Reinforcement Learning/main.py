from environments import EnvironmentInitializer
from argumentParser import ArgumentParser


def main():
    '''
    Main function to run the program:
    This function will create the pipeline to train the agent
    '''
    # Parse command line arguments
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    # Create environment initializer
    env_initializer = EnvironmentInitializer(env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.monitor_dir)



if __name__ == '__main__':
    main()
