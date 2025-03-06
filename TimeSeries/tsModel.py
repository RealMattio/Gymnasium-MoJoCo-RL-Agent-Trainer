import tensorflow as tf
import itertools
import numpy as np
from datasetModeler import DatasetModeller as dm
import os
from plotter import Plotter
import time

class TSModel:
    PARAM_GRID = {
        "lstm_units": [32, 64, 128],
        "dropout_rate": [0.0, 0.2],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [16, 32, 64],
        "epochs": [100, 50]  
    }
    @staticmethod
    def build_model(input_shape, lstm_units=64, dense_units=12, dropout_rate=0.0, learning_rate=0.001):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(lstm_units, dropout=dropout_rate, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(dense_units)
        ])
        
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        return model

    @staticmethod
    def load_model(model_path:str):
        return tf.keras.models.load_model(model_path)
    
    @staticmethod
    def GridSearch(X_trainval, y_trainval, dense_units:int=12, early_stopping:tf.keras.callbacks.EarlyStopping=None, param_grid:dict=PARAM_GRID):
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        best_score = -np.inf
        best_params = None
        input_shape = (X_trainval.shape[1], X_trainval.shape[2])

        for params in param_combinations:
            print(f"\nTesting parameters: {params}")
            try:
                model = TSModel.build_model(
                    input_shape=input_shape,
                    lstm_units=params['lstm_units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate'],
                    dense_units=dense_units
                )
                
                history = model.fit(
                    X_trainval, y_trainval,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    callbacks=[early_stopping],
                    validation_split=0.2,
                    verbose=2
                )
                
                val_loss = np.min(history.history['val_loss']) 
                current_score = -val_loss
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
                    print(f"Nuovo miglior punteggio: {best_score:.4f}")
                    
            except Exception as e:
                print(f"Fallito con parametri {params}: {str(e)}")
                continue

        print("\nMigliori iperparametri:", best_params)
        print("Miglior punteggio (neg_MSE):", best_score)
        return best_params, best_score
    
    @staticmethod
    def train_model(input_shape, best_params:dict, train_data, val_data, test_data, dense_units, early_stopping:tf.keras.callbacks.EarlyStopping=None ,shift:int=12, 
                    label_width:int=12, input_width:int=24, target_col:str="Potenza Uffici [W]", plot:bool=True, model_name:str=None, x_label:str=None, y_label:str=None,
                    num_subplots:int=1):
        final_model = TSModel.build_model(input_shape=input_shape,
            lstm_units=best_params["lstm_units"],
            dropout_rate=best_params["dropout_rate"],
            learning_rate=best_params["learning_rate"],
            dense_units=dense_units
        )
        train_ds = dm.make_dataset(train_data, input_width, label_width, shift, best_params["batch_size"], target_col=target_col)
        val_ds   = dm.make_dataset(val_data,   input_width, label_width, shift, best_params["batch_size"], target_col=target_col)
        test_ds  = dm.make_dataset(test_data,  input_width, label_width, shift, best_params["batch_size"], target_col=target_col)

        history = final_model.fit(
            train_ds,
            epochs=best_params["epochs"]+300,
            validation_data=val_ds,
            verbose=1,
            callbacks=[early_stopping]
        )
        test_loss, test_mae = final_model.evaluate(test_ds, verbose=1)
        print("Test loss:", test_loss, "Test MAE:", test_mae)
    
        test_predictions = final_model.predict(test_ds)
        test_labels = dm.extract_labels(test_ds)

        plotter = Plotter(test_labels, test_predictions, x_label, y_label, model_name, plot)
        plotter.test_plot(num_subplots)
        plotter.history_plot(history)
        return final_model, test_loss, test_mae
    
    @staticmethod
    def save_model(model, name:str ,path:str="./TimeSeries/models"):
        if not os.path.exists(path):
            os.makedirs(path)
        model.save(f"{path}/{name}.h5")
        print(f"Model saved in {path}/{name}.h5")

    @staticmethod
    def autoregressive_forecast(forecast_data, model, target_name:str, input_width:int=24):
        predictions = []
        ground_truth = []
        for i in range(input_width, len(forecast_data)):
            window = forecast_data.iloc[i - input_width:i].values
            window = window.reshape((1, input_width, forecast_data.shape[1]))
            pred_uff = model.predict(window, verbose=0)
            predictions.append(pred_uff[0, 0])
            ground_truth.append(forecast_data.iloc[i][target_name])
        return np.array(predictions), np.array(ground_truth)
    
    @staticmethod
    def forecast(forecast_data, model, target_name:str, label_width:int=12, input_width:int=24):
        predictions = []
        ground_truth = []

        for i in range(input_width, len(forecast_data) - label_width + 1):
            window = forecast_data.iloc[i - input_width:i].values
            window = window.reshape((1, input_width, forecast_data.shape[1]))
            pred_uff = model.predict(window, verbose=0)
            predictions.append(pred_uff[0])
            ground_truth.append(forecast_data.iloc[i : i + label_width][target_name].values)
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        full_pred = np.full(len(forecast_data), np.nan)
        full_gt = forecast_data[target_name].values

        num_samples = predictions.shape[0]

        for i_input in range(num_samples):
            i = input_width + i_input
            for step in range(label_width):
                t = i + step
                full_pred[t] = predictions[i_input, step]

        return full_pred, full_gt
    
    @staticmethod
    def autoregressive_forecast_full(forecast_data, model, target_name:str, input_width:int=24, forecast_steps:int=1440):
        full_data = forecast_data.values
        N, n_features = full_data.shape

        target_index = list(forecast_data.columns).index(target_name)
        
        init_start = N - forecast_steps - input_width
        init_end = N - forecast_steps
        current_window = full_data[init_start:init_end].copy()

        predictions = []
        for i in range(forecast_steps):
            input_window = current_window.reshape((1, input_width, n_features))
            pred = model.predict(input_window, verbose=0)
            predicted_value = pred[0, 0]
            predictions.append(predicted_value)
            
            new_row = current_window[-1].copy()
            new_row[target_index] = predicted_value
            current_window = np.vstack([current_window[1:], new_row])
        return np.array(predictions)
