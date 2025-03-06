import numpy as np
import os
import matplotlib.pyplot as plt
import random as rd

# Classe Plotter aggiornata
class Plotter:
    def __init__(self, real_values, forecasted_values, xlabel:str, ylabel:str, model_name:str, show_plot:bool, real_label='Real Values', forecasted_label='Forecasted Values'):
        """
        Crea un oggetto Plotter per visualizzare i valori reali e previsti.
        """
        self.real_values = real_values
        self.forecasted_values = forecasted_values
        self.real_label = real_label
        self.forecasted_label = forecasted_label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.model_name = model_name
        self.show_plot = show_plot

    def test_plot(self, num_subplots=3):
        """
        Plot di num_subplots intervalli (giornate) con 24 punti ciascuno.
        Viene selezionato un indice casuale che garantisce che la slice di real_values abbia esattamente 24 elementi.
        """
        # Converto i dati in array 1D
        real_vals = np.array(self.real_values).squeeze()
        forecast_vals = np.array(self.forecasted_values).squeeze()
        total_points = len(real_vals)
        if total_points < 24:
            raise ValueError("Not enough data points for plotting.")
        # Garantisco che random_index+24 sia <= total_points
        max_start = total_points - 24
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots))
        if num_subplots == 1:
            axes = [axes]
        for i in range(num_subplots):
            random_index = rd.randint(24, max_start)
            x_vals = np.arange(24)
            y_vals = real_vals[random_index:random_index+24]
            axes[i].plot(x_vals, y_vals, label='Data', color='blue', marker='.', linestyle='-')
            axes[i].set_title(f'{self.model_name} - Interval starting at index {random_index}')
            axes[i].set_xlabel(self.xlabel)
            axes[i].set_ylabel(self.ylabel)
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
        plt.tight_layout()
        if not os.path.exists('./TimeSeries/plots'):
            os.makedirs('./TimeSeries/plots')
        plt.savefig(f'./TimeSeries/plots/test_predictions_{self.model_name}.png', dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def points_plot(self, num_subplots=1, num_points=None):
        """
        Plot dei valori reali e previsti suddivisi in num_subplots segmenti.
        """
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
        plt.savefig(f'./TimeSeries/plots/forecasted_values_{self.model_name}.png', dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def history_plot(self, history):
        """
        Plot della storia dell'addestramento (loss e validation loss).
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
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
