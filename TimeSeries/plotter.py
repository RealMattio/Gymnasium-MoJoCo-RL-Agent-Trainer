import matplotlib.pyplot as plt
import tensorflow as tf
import os, math
import random as rd
import numpy as np
class Plotter:
    def __init__(self, real_values, forecasted_values, xlabel:str, ylabel:str, model_name:str, show_plot:bool,real_label='Real Values', forecasted_label='Forecasted Values'):
        '''
        create a Plotter object. it will be used to plot the real and forecasted values.
        :param real_values: The real values.
        :param forecasted_values: The forecasted values.
        :param xlabel: (str) The label of the x-axis.
        :param ylabel: (str) The label of the y-axis.
        :param model_name: (str) The name of the model.
        :param show_plot: (bool) If True the plot will be shown and saved. If False the plot will be saved only. 
        :param real_label: (str) The label of the real values.
        :param forecasted_label: (str) The label of the forecasted values.
        '''
        self.real_values = real_values
        self.forecasted_values = forecasted_values
        self.real_label = real_label
        self.forecasted_label = forecasted_label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.model_name = model_name
        self.show_plot = show_plot
    
    def test_plot(self, num_subplots=3):
        '''
        Plot num_subplots days of real and forecasted values. It will use the real values and forecasted values passed to the constructor.
        It will absume that are used 24 previous time stamp to forecast. Number of forecasted values is calculated from the shape of the forecasted values.
        :param num_subplots: (int) The number of subplots to create.
        '''
        # Ensure inputs are NumPy arrays
        #predictions = np.array(predictions)
        #ground_truth = np.array(ground_truth)
        
        # Check if the lengths of the time series match
        if len(self.forecasted_values) != len(self.real_values):
            raise ValueError("Predictions and ground truth must have the same length.")
        
        prevision_number, num_forecasted_values = self.forecasted_values.shape
        print(num_forecasted_values, prevision_number)
        timestamps = np.arange(24 + num_forecasted_values)
        # Create a figure with num_subplots subplots
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots))
        
        # If num_subplots is 1, axes is not a list, so we wrap it in a list for consistency
        if num_subplots == 1:
            axes = [axes]
        
        # Plot each interval
        for i in range(num_subplots):
            random_index = rd.randint(24,prevision_number-num_forecasted_values)
            
            if num_forecasted_values > 1:
                # plot dei dati usati per la previsione
                # calcolo quante previsioni sono necessarie per coprire 24 ore
                n = math.ceil(24/num_forecasted_values)
                # ottengo i valori del giorno prima che sono stati passati al modello per fare la previsione
                day_before = np.concatenate(self.real_values[random_index-n:random_index,:], axis=0)
                if len(day_before) > 24:
                    # elimino i dati in eccesso a partire dall'inizio
                    day_before = day_before[len(day_before)-24:]
                #plot dei dati reali del giorno prima
                axes[i].plot(timestamps[:24], day_before, label = 'Data', color = 'blue', marker='.', linestyle='-')
                #plot delle previsioni
                axes[i].scatter(timestamps[24:], self.forecasted_values[random_index,:], label = 'Predictions', color='red', marker='x')
                #plot della ground truth
                axes[i].scatter(timestamps[24:], self.real_values[random_index,:], label = 'Real Values', color='green', marker='o')
            else:
                axes[i].plot(timestamps[:24], self.real_values[random_index:random_index+24], label = 'Data', color = 'blue', marker='.', linestyle='-')
                axes[i].scatter(timestamps[24:], self.forecasted_values[random_index+24:random_index+24+num_forecasted_values], label = 'Predictions', color='red', marker='x')
                axes[i].scatter(timestamps[24:], self.real_values[random_index+24:random_index+24+num_forecasted_values], label = 'Real Values', color='green', marker='o')            
            # Add labels and legend
            axes[i].set_title(f'Test prediction model {self.model_name} - Interval {i + 1} (Timestamps {random_index} to {random_index+24+num_forecasted_values})')
            axes[i].set_xlabel(self.xlabel)
            axes[i].set_ylabel(self.ylabel)
            axes[i].legend(loc='upper left')
            axes[i].grid(True)
        
        # Adjust layout for better spacing
        plt.tight_layout()
        if not os.path.exists('./TimeSeries/plots'):
            os.makedirs('./TimeSeries/plots')
        plt.savefig(f'./TimeSeries/plots/test_predictions_{self.model_name}.png', dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def points_plot(self, num_subplots=1, num_points=None):
        '''
        Plot all the real and forecasted values divided in the number of subplot specified.
        :param num_subplots: (int) The number of subplots to create.
        :param num_points: (int) The number of points to plot.
        '''
        if num_points is None:
            num_points = len(self.real_values) // num_subplots
        
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))
        
        if num_subplots == 1:
            axes = [axes]
        
        total_points = len(self.real_values)
        segment_length = total_points // num_subplots
        
        for i in range(num_subplots):
            start_idx = i * segment_length
            end_idx = start_idx + num_points
            
            axes[i].plot(self.real_values[start_idx:end_idx], label=self.real_label, color='blue')
            axes[i].plot(self.forecasted_values[start_idx:end_idx], label=self.forecasted_label, color='red', linestyle='--')
            axes[i].set_xlabel(self.xlabel)
            axes[i].set_ylabel(self.ylabel)
            axes[i].set_title(f'Real vs Forecasted Values (Segment {i+1}) - Model {self.model_name}')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
        
        plt.tight_layout()
        if not os.path.exists('./TimeSeries/plots'):
            os.makedirs('./TimeSeries/plots')
        plt.savefig(f'./TimeSeries/plots/forcasted_values_{self.model_name}.png', dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        else:
            plt.close()
        
    def history_plot(self, history:tf.keras.callbacks.History):
        '''
        Plot the history of the model.
        :param model_name: (str) The name of the model.
        '''
        self.history = history
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss', color='blue')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} Model History')
        plt.legend(loc='upper right')
        plt.grid(True)
        if not os.path.exists('./TimeSeries/plots'):
            os.makedirs('./TimeSeries/plots')
        plt.savefig(f'./TimeSeries/plots/training_history_{self.model_name}.png', dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        else:
            plt.close()
        