from preprocessor import Preprocessor
from datasetModeler import DatasetModeller as dm
from tsModel import TSModel
from tensorflow.keras.callbacks import EarlyStopping
from plotter import Plotter
import numpy as np
import pandas as pd
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(description='Train and forecast parameters')
    parser.add_argument('--notrain', action='store_false', help='Hyperparameter tuning and training will not be performed')
    parser.add_argument('--nohptuning', action='store_false', help='Hyperparameter tuning will not be performed. Training will conintue with best parameters found by the authors')
    parser.add_argument('--noforecast', action='store_false', help='Forecasting will not be performed')
    parser.add_argument('--trainpath', type=str, default='TimeSeries/Dataset-Project-Deep-Learning-SMRES-Unificato.xlsx', help='Path to the dataset used for training')
    parser.add_argument('--save', action='store_true', help='If will be performed, save the new trained models into ./models folder')
    parser.add_argument('--trainplot', action='store_true', help='It will not be possible to view the five prediction plots on the test set')
    parser.add_argument('--forecastpath', type=str, default='TimeSeries/Dataset-Project-Deep-Learning-SMRES-Scartati.xlsx', help='Path to the dataset used for forecasting')
    parser.add_argument('--pretrained', action='store_true', help='Use the pretrained models') 
    args = parser.parse_args()
    TRAIN_MEAN = None
    TRAIN_STD = None
    if args.notrain:
        if args.trainpath is not None:
            TRAIN_DATA_PATH = args.trainpath
        else:
            TRAIN_DATA_PATH = 'TimeSeries/Dataset-Project-Deep-Learning-SMRES-Unificato.xlsx'
        train_preprocessor = Preprocessor(data_path=TRAIN_DATA_PATH)
        train_preprocessor.load_data()
        preprocessed_train_data = train_preprocessor.preprocess_data()
        train_data, val_data, test_data = train_preprocessor.divide_train_val_test_data()
        TRAIN_MEAN = train_preprocessor.train_mean
        TRAIN_STD = train_preprocessor.train_std

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        # Creazione delle sequenze per la grid search per il modello di previsione degli uffici (12h)
        X_train_uff_12h, y_train_uff_12h = dm.create_sequences_df(train_data, input_width=24, out_steps=12, target_col="Potenza Uffici [W]")
        X_val_uff_12h,   y_val_uff_12h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=12, target_col="Potenza Uffici [W]")

        # Uniamo training e validation per la grid search
        X_trainval_uff_12h = np.concatenate([X_train_uff_12h, X_val_uff_12h], axis=0)
        y_trainval_uff_12h = np.concatenate([y_train_uff_12h, y_val_uff_12h], axis=0)
    
        # Creazione delle sequenze per il modello di previsione degli uffici (1h)
        X_train_uff_1h, y_train_uff_1h = dm.create_sequences_df(train_data, input_width=24, out_steps=1, target_col="Potenza Uffici [W]")
        X_val_uff_1h,   y_val_uff_1h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=1, target_col="Potenza Uffici [W]")

        # Uniamo training e validation per la grid search
        X_trainval_uff_1h = np.concatenate([X_train_uff_1h, X_val_uff_1h], axis=0)
        y_trainval_uff_1h = np.concatenate([y_train_uff_1h, y_val_uff_1h], axis=0)
        
        # Creazione delle sequenze per il modello di previsione dell'irraggiamento (12h)
        X_train_irr_12h, y_train_irr_12h = dm.create_sequences_df(train_data, input_width=24, out_steps=12, target_col="Irraggiamento [kWh/m2]")
        X_val_irr_12h,   y_val_irr_12h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=12, target_col="Irraggiamento [kWh/m2]")

        # Uniamo training e validation per la grid search
        X_trainval_irr_12h = np.concatenate([X_train_irr_12h, X_val_irr_12h], axis=0)
        y_trainval_irr_12h = np.concatenate([y_train_irr_12h, y_val_irr_12h], axis=0)
        # Creazione delle sequenze per il modello di previsione dell'irraggiamento (1h)
        X_train_irr_1h, y_train_irr_1h = dm.create_sequences_df(train_data, input_width=24, out_steps=1, target_col="Irraggiamento [kWh/m2]")
        X_val_irr_1h,   y_val_irr_1h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=1, target_col="Irraggiamento [kWh/m2]")

        # Uniamo training e validation per la grid search
        X_trainval_irr_1h = np.concatenate([X_train_irr_1h, X_val_irr_1h], axis=0)
        y_trainval_irr_1h = np.concatenate([y_train_irr_1h, y_val_irr_1h], axis=0)

        if args.nohptuning:
            # GRID SEARCH DEI QUATTRO MODELLI
            # Eseguiamo la grid search
            best_params_uff_12h, best_score_uff_12h = TSModel.GridSearch(X_trainval_uff_12h, y_trainval_uff_12h, dense_units=12, early_stopping=early_stopping)
            print(f"\nHyperparameter tuning for uffici 12h model completed.\nBest parameters: {best_params_uff_12h}, Best score: {best_score_uff_12h}")

            best_params_uff_1h, best_score_uff_1h = TSModel.GridSearch(X_trainval_uff_1h, y_trainval_uff_1h, dense_units=1, early_stopping=early_stopping)
            print(f"\nHyperparameter tuning for uffici 1h model completed.\nBest parameters: {best_params_uff_1h}, Best score: {best_score_uff_1h}")

            best_params_irr_12h, best_score_irr_12h = TSModel.GridSearch(X_trainval_irr_12h, y_trainval_irr_12h, dense_units=12, early_stopping=early_stopping)
            print(f"\nHyperparameter tuning for irraggiamento 12h model completed.\nBest parameters: {best_params_irr_12h}, Best score: {best_score_irr_12h}")

            best_params_irr_1h, best_score_irr_1h = TSModel.GridSearch(X_trainval_irr_1h, y_trainval_irr_1h, dense_units=1, early_stopping=early_stopping)
            print(f"\nHyperparameter tuning for irraggiamento 1h model completed.\nBest parameters: {best_params_irr_1h}, Best score: {best_score_irr_1h}")
        else:
            # Caricamento dei migliori parametri trovati dagli autori
            with open('TimeSeries/pretrained_models/best_params.json') as f:
                best_params = json.load(f)
            best_params_uff_12h = best_params['uffici_12h']
            best_params_uff_1h = best_params['uffici_1h']
            best_params_irr_12h = best_params['irraggiamento_12h']
            best_params_irr_1h = best_params['irraggiamento_1h']
            
        # ADDESTRAMENTO DEI MODELLI
        # Addestramento del modello di previsione degli uffici (12h)
        uffici_12h_model, uffici_12h_test_loss, uffici_12h_test_mae = TSModel.train_model(
            input_shape=(X_train_uff_12h.shape[1], X_train_uff_12h.shape[2]),
            best_params=best_params_uff_12h,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            dense_units=12,
            early_stopping=early_stopping,
            shift=12,
            label_width=12,
            input_width=24,
            target_col="Potenza Uffici [W]",
            plot=args.trainplot,
            model_name="Uffici 12h",
            x_label="Timestamp",
            y_label="Previsione uffici standardizzata",
            num_subplots=3
        )
        print(f"\nUffici 12h model training completed.\nTest loss: {uffici_12h_test_loss}, Test MAE: {uffici_12h_test_mae}")
        #salvataggio del modello
        if args.save:
            TSModel.save_model(uffici_12h_model, "uffici_12h")
        
        # Addestramento del modello di previsione degli uffici (1h)
        uffici_1h_model, uffici_1h_test_loss, uffici_1h_test_mae = TSModel.train_model(
            input_shape=(X_train_uff_1h.shape[1], X_train_uff_1h.shape[2]),
            best_params=best_params_uff_1h,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            dense_units=1,
            early_stopping=early_stopping,
            shift=1,
            label_width=1,
            input_width=24,
            target_col="Potenza Uffici [W]",
            plot=args.trainplot,
            model_name="Uffici 1h",
            x_label="Timestamp",
            y_label="Previsione uffici standardizzata",
            num_subplots=3
        )
        print(f"\nUffici 1h model training completed.\nTest loss: {uffici_1h_test_loss}, Test MAE: {uffici_1h_test_mae}")
        #salvataggio del modello
        if args.save:
            TSModel.save_model(uffici_1h_model, "uffici_1h")
        
        # Addestramento del modello di previsione dell'irraggiamento (12h)
        irr_12h_model, irr_12h_test_loss, irr_12h_test_mae = TSModel.train_model(
            input_shape=(X_train_irr_12h.shape[1], X_train_irr_12h.shape[2]),
            best_params=best_params_irr_12h,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            dense_units=12,
            early_stopping=early_stopping,
            shift=12,
            label_width=12,
            input_width=24,
            target_col="Irraggiamento [kWh/m2]",
            plot=args.trainplot,
            model_name="Irraggiamento 12h",
            x_label="Timestamp",
            y_label="Previsione irraggiamento standardizzata",
            num_subplots=3
        )
        print(f"\nIrraggiamento 12h model training completed.\nTest loss: {irr_12h_test_loss}, Test MAE: {irr_12h_test_mae}")
        #salvataggio del modello
        if args.save:
            TSModel.save_model(irr_12h_model, "irraggiamento_12h")
        
        # Addestramento del modello di previsione dell'irraggiamento (1h)
        irr_1h_model, irr_1h_test_loss, irr_1h_test_mae = TSModel.train_model(
            input_shape=(X_train_irr_1h.shape[1], X_train_irr_1h.shape[2]),
            best_params=best_params_irr_1h,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            dense_units=1,
            early_stopping=early_stopping,
            shift=1,
            label_width=1,
            input_width=24,
            target_col="Irraggiamento [kWh/m2]",
            plot=args.trainplot,
            model_name="Irraggiamento 1h",
            x_label="Timestamp",
            y_label="Previsione irraggiamento standardizzata",
            num_subplots=3
        )
        print(f"\nIrraggiamento 1h model training completed.\nTest loss: {irr_1h_test_loss}, Test MAE: {irr_1h_test_mae}")
        #salvataggio del modello
        if args.save:
            TSModel.save_model(irr_1h_model, "irraggiamento_1h")
    
    if args.noforecast:
        # CARICAMENTO DEI MODELLI
        if args.pretrained:
            try:
                uffici_12h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_uff.h5")
                uffici_1h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_uff_autoreg.h5")
                irr_12h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_irr.h5")
                irr_1h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_irr_autoreg.h5")
            except FileNotFoundError:
                raise FileNotFoundError("The models are not available. Please check pretrained models are in './pretrained_models' folder. If not download them first from Git")
        else:
            try:
                uffici_12h_model = TSModel.load_model("./TimeSeries/models/uffici_12h.h5")
                uffici_1h_model = TSModel.load_model("./TimeSeries/models/uffici_1h.h5")
                irr_12h_model = TSModel.load_model("./TimeSeries/models/irraggiamento_12h.h5")
                irr_1h_model = TSModel.load_model("./TimeSeries/models/irraggiamento_1h.h5")
            except FileNotFoundError:
                raise FileNotFoundError("The models are not available. Please check models are in './models' folder. If not train new models first ore use pretrained models.")
        if TRAIN_MEAN is None or TRAIN_STD is None:
            # Se la media e la deviazione standard non sono disponibili vuol dire che l'utente ha scelto di non eseguire il training. Pertanto, carichiamo i dati di training e calcoliamo media e deviazione standard
            if args.trainpath is not None:
                TRAIN_DATA_PATH = args.trainpath
            else:
                TRAIN_DATA_PATH = 'TimeSeries/Dataset-Project-Deep-Learning-SMRES-Unificato.xlsx'
            train_preprocessor = Preprocessor(data_path=TRAIN_DATA_PATH)
            train_preprocessor.load_data()
            preprocessed_train_data = train_preprocessor.preprocess_data()
            train_data, val_data, test_data = train_preprocessor.divide_train_val_test_data()
            TRAIN_MEAN = train_preprocessor.train_mean
            TRAIN_STD = train_preprocessor.train_std
        
        # Caricamento e preprocessamento dei dati che verranno usati per il forecasting
        if args.forecastpath is not None:
            FORECAST_DATA_PATH = args.forecastpath
        else:
            FORECAST_DATA_PATH = 'TimeSeries/Dataset-Project-Deep-Learning-SMRES-Scartati.xlsx'
        forecast_preprocessor = Preprocessor(data_path=FORECAST_DATA_PATH)
        forecast_preprocessor.load_data() # Caricamento dei dati
        preprocessed_forecast_data = forecast_preprocessor.preprocess_data() # Preprocessamento dei dati
        forecast_preprocessor.standardize_data(TRAIN_MEAN, TRAIN_STD) # Standardizzazione dei dati
        preprocessed_forecast_data = forecast_preprocessor.data # Aggiornamento dei dati standardizzati

        # Possiamo procedere con il forecasting dei dati dei quattro modelli
        # Il primo ciclo servirà per i modelli a 1h (Autoregressivi)
        uffici_1h_forecast, uffici_1h_ground_truth = TSModel.autoregressive_forecast(preprocessed_forecast_data, uffici_1h_model, "Potenza Uffici [W]")
        print(f"\nUffici 1h model forecast completed.")
        irr_1h_forecast, irr_1h_ground_truth = TSModel.autoregressive_forecast(preprocessed_forecast_data, irr_1h_model, "Irraggiamento [kWh/m2]")
        print(f"\nIrraggiamento 1h model forecast completed.")
        uffici_1h_mae = np.mean(np.abs(np.array(uffici_1h_forecast) - np.array(uffici_1h_ground_truth)))
        irr_1h_mae = np.mean(np.abs(np.array(irr_1h_forecast) - np.array(irr_1h_ground_truth)))

        # Il secondo ciclo servirà per i modelli a 12h
        uffici_12h_forecast, uffici_12h_ground_truth = TSModel.forecast(preprocessed_forecast_data, uffici_12h_model, "Potenza Uffici [W]")
        print(f"\nUffici 12h model forecast completed.")
        irr_12h_forecast, irr_12h_ground_truth = TSModel.forecast(preprocessed_forecast_data, irr_12h_model, "Irraggiamento [kWh/m2]")
        print(f"\nIrraggiamento 12h model forecast completed.")
        uffici_12h_mae = np.mean(np.abs(np.array(uffici_12h_forecast) - np.array(uffici_12h_ground_truth)))
        irr_12h_mae = np.mean(np.abs(np.array(irr_12h_forecast) - np.array(irr_12h_ground_truth)))
        
        uffici_1h_plotter = Plotter(uffici_1h_ground_truth, uffici_1h_forecast, "Timestamp", "Potenza Uffici Standardizzata", 'Uffici 1h forecast')
        uffici_1h_plotter.points_plot(num_subplots=3)

        irr_1h_plotter = Plotter(irr_1h_ground_truth, irr_1h_forecast, "Timestamp", "Irraggiamento Standardizzato", 'Irraggiamento 1h forecast')
        irr_1h_plotter.points_plot(num_subplots=3)

        uffici_12h_plotter = Plotter(uffici_12h_ground_truth, uffici_12h_forecast, "Timestamp", "Potenza Uffici Standardizzata", 'Uffici 12h forecast')
        uffici_12h_plotter.points_plot(num_subplots=3)

        irr_12h_plotter = Plotter(irr_12h_ground_truth, irr_12h_forecast, "Timestamp", "Irraggiamento Standardizzato", 'Irraggiamento 12h forecast')
        irr_12h_plotter.points_plot(num_subplots=3)

        # salvo i dati predetti nella cartella forecasted_data
        if not os.path.exists('TimeSeries/forecasted_data'):
            os.makedirs('TimeSeries/forecasted_data')
        pd.DataFrame(uffici_1h_forecast).to_excel('TimeSeries/forecasted_data/uffici_1h_forecast.xlsx', index=False)
        pd.DataFrame(irr_1h_forecast).to_excel('TimeSeries/forecasted_data/irraggiamento_1h_forecast.xlsx', index=False)
        pd.DataFrame(uffici_12h_forecast).to_excel('TimeSeries/forecasted_data/uffici_12h_forecast.xlsx', index=False)
        pd.DataFrame(irr_12h_forecast).to_excel('TimeSeries/forecasted_data/irraggiamento_12h_forecast.xlsx', index=False)
        print(f"\nForecasted data saved.")
    
    print("Training and forecasting completed.")
        
if __name__ == "__main__":
    main()