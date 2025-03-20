import matplotlib.pyplot as plt
import os
import numpy as np

class Plotter:

    @staticmethod
    def box_plot(data_list:list, algorithm_names:list, title:str, xlabel:str, ylabel:str, save_path:str="./results/plots", save=True):
        """
        Generates a box plot from a list of data points and corresponding labels.

        Parameters:
        - data_list: list of lists, where each inner list contains data points for a specific algorithm.
        - algorithm_names: list of str, names of the algorithms corresponding to the data.
        - title: str, title of the plot.
        - xlabel: str, label for the x-axis.
        - ylabel: str, label for the y-axis.
        """
        plt.figure(figsize=(10, 6))
        boxplot = plt.boxplot(data_list, labels=algorithm_names, patch_artist=True)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(algorithm_names)))
        for box, color in zip(boxplot['boxes'], colors):
            box.set(facecolor=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{title}.png")
        plt.show()

    @staticmethod
    def plot_rewards(rewards_list:list, model_names:list, title:str, xlabel:str, ylabel:str, save_path:str="./results/plots", save=True):
        """
        Plots the rewards obtained during episodes for multiple models.

        Parameters:
        - rewards_list: 2D list, where each inner list contains the rewards for a specific model.
        - model_names: list of str, names of the models corresponding to the rewards.
        """
        plt.figure(figsize=(10, 6))
        for rewards, model_name in zip(rewards_list, model_names):
            plt.plot(rewards, label=model_name, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/rewards_per_episode.png")
        plt.show()