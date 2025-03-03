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
        '''
        Create and compile a LSTM model
        :param input_shape: (tuple) The shape of the input.
        :param lstm_units: (int) The number of LSTM units.
        :param dense_units: (int) The number of units in the Dense layer.
        :param dropout_rate: (float) The dropout rate.
        :param learning_rate: (float) The learning rate.
        :return: (tf.keras.Model) The LSTM model.
        '''
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            #tf.keras.layers.LSTM(lstm_units*2, dropout=dropout_rate, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            #tf.keras.layers.Dropout(0.3),
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
        '''
        Load a Keras model from a file.
        :param model_path: (str) The path to the model file.
        :return: (tf.keras.Model) The loaded model.
        '''
        return tf.keras.models.load_model(model_path)
    
    @staticmethod
    def GridSearch(X_trainval, y_trainval, dense_units:int=12, early_stopping:tf.keras.callbacks.EarlyStopping=None, param_grid:dict=PARAM_GRID):
        '''
        Perform a grid search to find the best hyperparameters for the LSTM model.
        :param X_trainval: (np.array) The training and validation input data.
        :param y_trainval: (np.array) The training and validation target data.
        :param dense_units: (int) The number of units in the Dense layer. It's equal to the number of hour to forecast.
        :param early_stopping: (tf.keras.callbacks.EarlyStopping) The early stopping callback.
        :param param_grid: (dict) The hyperparameters grid.
        :return: (dict, float) The best hyperparameters and the best score.
        '''
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        best_score = -np.inf
        best_params = None
        input_shape = (X_trainval.shape[1], X_trainval.shape[2])  # Forma dell'input

        for params in param_combinations:
            print(f"\nTesting parameters: {params}")
            
            try:
                # Costruisci e allena il modello
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
                    validation_split=0.2,  # Sostituisci con la tua strategia di validazione
                    verbose=2
                )
                
                # Callcolo dello score migliore prendendo il minor valore di loss registrato (usiamo la validation loss)
                val_loss = np.min(history.history['val_loss']) 
                current_score = -val_loss  # Negativo per mantenere la stessa metrica
                
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
        '''
        Train the final model using the best hyperparameters found with the grid search.
        :param input_shape: (tuple) The shape of the input.
        :param best_params: (dict) The best hyperparameters found with the grid search.
        :param train_data: (pd.DataFrame) The training data.
        :param val_data: (pd.DataFrame) The validation data.
        :param test_data: (pd.DataFrame) The test data.
        :param early_stopping: (tf.keras.callbacks.EarlyStopping) The early stopping callback.
        :param shift: (int) The number of steps to shift the target.
        :param label_width: (int) The width of the label.
        :param input_width: (int) The width of the input.
        :param target_col: (str) The target column.
        :param plot: (bool) Whether to plot the results.
        :param model_name: (str) The name of the model.
        :param x_label: (str) The label of the x-axis.
        :param y_label: (str) The label of the y-axis.
        :param num_subplots: (int) The number of subplots to create.
        :return: (tf.keras.Model, float, float) The trained model, the test loss and the test MAE.
        '''
        final_model = TSModel.build_model(input_shape=input_shape,
            lstm_units=best_params["lstm_units"],
            dropout_rate=best_params["dropout_rate"],
            learning_rate=best_params["learning_rate"],
            dense_units=dense_units  # questo parametro corrisponde al numero di uscite e quindi alle ore che devono essere predette
        )
        # Ricreiamo i dataset TF con il batch size ottimale dalla grid search
        # La funzione make_dataset ha come paramertro di default target_col impostato sulla colonna uffici
        train_ds = dm.make_dataset(train_data, input_width, label_width, shift, best_params["batch_size"], target_col=target_col)
        val_ds   = dm.make_dataset(val_data, input_width, label_width, shift, best_params["batch_size"], target_col=target_col)
        test_ds  = dm.make_dataset(test_data, input_width, label_width, shift, best_params["batch_size"], target_col=target_col)

        # 8. Addestramento del modello finale
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
        # stampo le dimensioi di test predictions e test labels
        # print(test_predictions.shape)
        # print(test_labels.shape)
        # time.sleep(1)
        plotter = Plotter(test_labels, test_predictions, x_label, y_label, model_name, plot)
        plotter.test_plot(num_subplots)
        plotter.history_plot(history)
        return final_model, test_loss, test_mae
    
    @staticmethod
    def save_model(model, name:str ,path:str="./TimeSeries/models"):
        '''
        Save a Keras model to a file.
        :param model: (tf.keras.Model) The model to save.
        :param name: (str) The name of the model.
        :param path: (str) The path to save the model. It mustn't include the name of the model.
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        # salvo il modello addestrato
        model.save(f"{path}/{name}.h5")
        print(f"Model saved in {path}/{name}.h5")

    @staticmethod
    def autoregressive_forecast(forecast_data, model, target_name:str, input_width:int=24):
        '''
        Perform an autoregressive forecast.
        '''
        predictions = []
        ground_truth = []

        # Ciclo autoregressivo one-step ahead
        for i in range(input_width, len(forecast_data)):
            # Estrazione della finestra di input: ultime 24 righe dai dati (già standardizzati)
            window = forecast_data.iloc[i - input_width:i].values
            window = window.reshape((1, input_width, forecast_data.shape[1]))
            
            # Previsione per target_name con il modello autoregressivo
            pred_uff = model.predict(window, verbose=0)
            predictions.append(pred_uff[0, 0])
            
            # Ground truth (già standardizzata) per la stessa riga
            ground_truth.append(forecast_data.iloc[i][target_name])
        # Converto le liste in array numpy
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        return predictions, ground_truth
    
    @staticmethod
    def forecast(forecast_data, model, target_name:str, label_width:int=12, input_width:int=24):
        '''
        Perform a forecast.
        '''
        predictions = []
        ground_truth = []

        # Ciclo autoregressivo one-step ahead
        for i in range(input_width, len(forecast_data) - label_width + 1):
            # Estrazione della finestra di input: ultime 24 righe dai dati (già standardizzati)
            window = forecast_data.iloc[i - input_width:i].values
            window = window.reshape((1, input_width, forecast_data.shape[1]))
            
            # Previsione per target_name con il modello autoregressivo
            pred_uff = model.predict(window, verbose=0)
            predictions.append(pred_uff[0])
            
            # Ground truth (già standardizzata) per la stessa riga
            ground_truth.append(forecast_data.iloc[i : i + label_width][target_name].values)
        # converto le liste in array numpy
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        '''
        This process ensures that the predictions are placed at the correct time steps in the `full_pred`
        array, aligning them with the ground truth values for subsequent analysis and visualization.
        '''
        full_pred = np.full(len(forecast_data), np.nan)  # array vuoto di dimensione "length"
        full_gt = forecast_data[target_name].values

        num_samples = predictions.shape[0]

        for i_input in range(num_samples):
            i = input_width + i_input
            for step in range(label_width):  # 0..11
                t = i + step
                full_pred[t] = predictions[i_input, step]

        return full_pred, full_gt

    