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
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Train and forecast parameters')
    parser.add_argument('--notrain', action='store_false', help='Hyperparameter tuning and training will not be performed')
    parser.add_argument('--nohptuning', action='store_false', help='Hyperparameter tuning will not be performed. Training will continue with best parameters found by the authors')
    parser.add_argument('--noforecast', action='store_false', help='Forecasting will not be performed')
    parser.add_argument('--trainpath', type=str, default='TimeSeries/Dataset-Project-Deep-Learning-SMRES-Unificato.xlsx', help='Path to the dataset used for training')
    parser.add_argument('--save', action='store_true', help='If set, save the newly trained models into the TimeSeries/models folder')
    parser.add_argument('--trainplot', action='store_true', help='If set, show prediction plots during training; otherwise, they are saved automatically in TimeSeries/plots')
    parser.add_argument('--forecastplot', action='store_true', help='If set, show forecast plots')
    parser.add_argument('--forecastpath', type=str, default='TimeSeries/Dataset-Project-Deep-Learning-SMRES-Scartati.xlsx', help='Path to the dataset used for forecasting')
    parser.add_argument('--pretrained', action='store_true', help='Use the pretrained models') 
    args = parser.parse_args()

    TRAIN_MEAN = None
    TRAIN_STD = None

    # ----------------- Training Section -----------------
    if args.notrain:
        TRAIN_DATA_PATH = args.trainpath if args.trainpath is not None else 'TimeSeries/Dataset-Project-Deep-Learning-SMRES-Unificato.xlsx'
        train_preprocessor = Preprocessor(data_path=TRAIN_DATA_PATH)
        train_preprocessor.load_data()
        preprocessed_train_data = train_preprocessor.preprocess_data()
        train_data, val_data, test_data = train_preprocessor.divide_train_val_test_data()
        TRAIN_MEAN = train_preprocessor.train_mean
        TRAIN_STD = train_preprocessor.train_std

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        # Sequenze per Uffici (12h)
        X_train_uff_12h, y_train_uff_12h = dm.create_sequences_df(train_data, input_width=24, out_steps=12, target_col="Potenza Uffici [W]")
        X_val_uff_12h,   y_val_uff_12h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=12, target_col="Potenza Uffici [W]")
        X_trainval_uff_12h = np.concatenate([X_train_uff_12h, X_val_uff_12h], axis=0)
        y_trainval_uff_12h = np.concatenate([y_train_uff_12h, y_val_uff_12h], axis=0)
    
        # Sequenze per Uffici (1h)
        X_train_uff_1h, y_train_uff_1h = dm.create_sequences_df(train_data, input_width=24, out_steps=1, target_col="Potenza Uffici [W]")
        X_val_uff_1h,   y_val_uff_1h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=1, target_col="Potenza Uffici [W]")
        X_trainval_uff_1h = np.concatenate([X_train_uff_1h, X_val_uff_1h], axis=0)
        y_trainval_uff_1h = np.concatenate([y_train_uff_1h, y_val_uff_1h], axis=0)
        
        # Sequenze per Irraggiamento (12h)
        X_train_irr_12h, y_train_irr_12h = dm.create_sequences_df(train_data, input_width=24, out_steps=12, target_col="Irraggiamento [kWh/m2]")
        X_val_irr_12h,   y_val_irr_12h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=12, target_col="Irraggiamento [kWh/m2]")
        X_trainval_irr_12h = np.concatenate([X_train_irr_12h, X_val_irr_12h], axis=0)
        y_trainval_irr_12h = np.concatenate([y_train_irr_12h, y_val_irr_12h], axis=0)
        # Sequenze per Irraggiamento (1h)
        X_train_irr_1h, y_train_irr_1h = dm.create_sequences_df(train_data, input_width=24, out_steps=1, target_col="Irraggiamento [kWh/m2]")
        X_val_irr_1h,   y_val_irr_1h   = dm.create_sequences_df(val_data,   input_width=24, out_steps=1, target_col="Irraggiamento [kWh/m2]")
        X_trainval_irr_1h = np.concatenate([X_train_irr_1h, X_val_irr_1h], axis=0)
        y_trainval_irr_1h = np.concatenate([y_train_irr_1h, y_val_irr_1h], axis=0)

        if args.nohptuning:
            best_params_uff_12h, best_score_uff_12h = TSModel.GridSearch(X_trainval_uff_12h, y_trainval_uff_12h, dense_units=12, early_stopping=early_stopping)
            print(f"\nUffici 12h tuning completed.\nBest parameters: {best_params_uff_12h}, Best score: {best_score_uff_12h}")
            best_params_uff_1h, best_score_uff_1h = TSModel.GridSearch(X_trainval_uff_1h, y_trainval_uff_1h, dense_units=1, early_stopping=early_stopping)
            print(f"\nUffici 1h tuning completed.\nBest parameters: {best_params_uff_1h}, Best score: {best_score_uff_1h}")
            best_params_irr_12h, best_score_irr_12h = TSModel.GridSearch(X_trainval_irr_12h, y_trainval_irr_12h, dense_units=12, early_stopping=early_stopping)
            print(f"\nIrraggiamento 12h tuning completed.\nBest parameters: {best_params_irr_12h}, Best score: {best_score_irr_12h}")
            best_params_irr_1h, best_score_irr_1h = TSModel.GridSearch(X_trainval_irr_1h, y_trainval_irr_1h, dense_units=1, early_stopping=early_stopping)
            print(f"\nIrraggiamento 1h tuning completed.\nBest parameters: {best_params_irr_1h}, Best score: {best_score_irr_1h}")
        else:
            with open('TimeSeries/pretrained_models/best_params.json') as f:
                best_params = json.load(f)
            best_params_uff_12h = best_params['uffici_12h']
            best_params_uff_1h = best_params['uffici_1h']
            best_params_irr_12h = best_params['irraggiamento_12h']
            best_params_irr_1h = best_params['irraggiamento_1h']
            
        # Training dei modelli
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
        print(f"\nUffici 12h training completed.\nTest loss: {uffici_12h_test_loss}, Test MAE: {uffici_12h_test_mae}")
        if args.save:
            TSModel.save_model(uffici_12h_model, "uffici_12h")
            try:
                with open('TimeSeries/models/best_params.json', 'r') as f:
                    best_param_file = json.load(f)
                best_param_file['uffici_12h'] = best_params_uff_12h
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump(best_param_file, f)
            except FileNotFoundError:
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump({'uffici_12h': best_params_uff_12h}, f)

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
        print(f"\nUffici 1h training completed.\nTest loss: {uffici_1h_test_loss}, Test MAE: {uffici_1h_test_mae}")
        if args.save:
            TSModel.save_model(uffici_1h_model, "uffici_1h")
            try:
                with open('TimeSeries/models/best_params.json', 'r') as f:
                    best_param_file = json.load(f)
                best_param_file['uffici_1h'] = best_params_uff_1h
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump(best_param_file, f)
            except FileNotFoundError:
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump({'uffici_1h': best_params_uff_1h}, f)
        
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
        print(f"\nIrraggiamento 12h training completed.\nTest loss: {irr_12h_test_loss}, Test MAE: {irr_12h_test_mae}")
        if args.save:
            TSModel.save_model(irr_12h_model, "irraggiamento_12h")
            try:
                with open('TimeSeries/models/best_params.json', 'r') as f:
                    best_param_file = json.load(f)
                best_param_file['irraggiamento_12h'] = best_params_irr_12h
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump(best_param_file, f)
            except FileNotFoundError:
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump({'irraggiamento_12h': best_params_irr_12h}, f)
        
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
        print(f"\nIrraggiamento 1h training completed.\nTest loss: {irr_1h_test_loss}, Test MAE: {irr_1h_test_mae}")
        if args.save:
            TSModel.save_model(irr_1h_model, "irraggiamento_1h")
            try:
                with open('TimeSeries/models/best_params.json', 'r') as f:
                    best_param_file = json.load(f)
                best_param_file['irraggiamento_1h'] = best_params_irr_1h
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump(best_param_file, f)
            except FileNotFoundError:
                with open('TimeSeries/models/best_params.json', 'w') as f:
                    json.dump({'irraggiamento_1h': best_params_irr_1h}, f)
        print("Training completed.")
    
    # ----------------- Forecasting Section -----------------
    if args.noforecast:
        if args.pretrained:
            try:
                uffici_12h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_uff.h5")
                uffici_1h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_uff_autoreg.h5")
                irr_12h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_irr.h5")
                irr_1h_model = TSModel.load_model("TimeSeries/pretrained_models/final_model_irr_autoreg.h5")
            except FileNotFoundError:
                raise FileNotFoundError("Pretrained models not found in './pretrained_models' folder.")
        else:
            try:
                uffici_12h_model = TSModel.load_model("./TimeSeries/models/uffici_12h.h5")
                uffici_1h_model = TSModel.load_model("./TimeSeries/models/uffici_1h.h5")
                irr_12h_model = TSModel.load_model("./TimeSeries/models/irraggiamento_12h.h5")
                irr_1h_model = TSModel.load_model("./TimeSeries/models/irraggiamento_1h.h5")
            except FileNotFoundError:
                raise FileNotFoundError("Models not found in './models' folder.")
        
        if TRAIN_MEAN is None or TRAIN_STD is None:
            TRAIN_DATA_PATH = args.trainpath if args.trainpath is not None else 'TimeSeries/Dataset-Project-Deep-Learning-SMRES-Unificato.xlsx'
            train_preprocessor = Preprocessor(data_path=TRAIN_DATA_PATH)
            train_preprocessor.load_data()
            preprocessed_train_data = train_preprocessor.preprocess_data()
            train_data, val_data, test_data = train_preprocessor.divide_train_val_test_data()
            TRAIN_MEAN = train_preprocessor.train_mean
            TRAIN_STD = train_preprocessor.train_std
        
        FORECAST_DATA_PATH = args.forecastpath if args.forecastpath is not None else 'TimeSeries/Dataset-Project-Deep-Learning-SMRES-Scartati.xlsx'
        forecast_preprocessor = Preprocessor(data_path=FORECAST_DATA_PATH)
        forecast_preprocessor.load_data()
        preprocessed_forecast_data = forecast_preprocessor.preprocess_data()
        forecast_preprocessor.standardize_data(TRAIN_MEAN, TRAIN_STD)
        preprocessed_forecast_data = forecast_preprocessor.data

        forecast_steps = 1440
        input_width = 24

        # Forecast autoregressivo per ciascun modello
        uffici_1h_forecast = TSModel.autoregressive_forecast_full(preprocessed_forecast_data, uffici_1h_model, "Potenza Uffici [W]", input_width, forecast_steps)
        irr_1h_forecast    = TSModel.autoregressive_forecast_full(preprocessed_forecast_data, irr_1h_model,    "Irraggiamento [kWh/m2]", input_width, forecast_steps)
        uffici_12h_forecast = TSModel.autoregressive_forecast_full(preprocessed_forecast_data, uffici_12h_model, "Potenza Uffici [W]", input_width, forecast_steps)
        irr_12h_forecast    = TSModel.autoregressive_forecast_full(preprocessed_forecast_data, irr_12h_model,    "Irraggiamento [kWh/m2]", input_width, forecast_steps)
        
        # Ground truth (standardizzata) per il calcolo metriche
        data_series_uff = preprocessed_forecast_data["Potenza Uffici [W]"].values
        uffici_ground_truth = data_series_uff[-forecast_steps:]
        data_series_irr = preprocessed_forecast_data["Irraggiamento [kWh/m2]"].values
        irr_ground_truth = data_series_irr[-forecast_steps:]

        # --- Calcolo metriche sui valori standardizzati ---
        def compute_nmae_by_hour(predictions, ground_truth):
            nmae_per_hour = []
            for hour in range(24):
                idx = np.where(np.arange(len(predictions)) % 24 == hour)[0]
                if len(idx) > 0:
                    mae = np.mean(np.abs(predictions[idx] - ground_truth[idx]))
                    norm = np.mean(np.abs(ground_truth[idx]))
                    nmae = mae / norm if norm != 0 else mae
                    nmae_per_hour.append(nmae)
                else:
                    nmae_per_hour.append(np.nan)
            return np.array(nmae_per_hour)

        def compute_rmse_by_hour(predictions, ground_truth):
            rmse_per_hour = []
            for hour in range(24):
                idx = np.where(np.arange(len(predictions)) % 24 == hour)[0]
                if len(idx) > 0:
                    rmse = np.sqrt(np.mean((predictions[idx] - ground_truth[idx])**2))
                    rmse_per_hour.append(rmse)
                else:
                    rmse_per_hour.append(np.nan)
            return np.array(rmse_per_hour)

        # NMAE Uffici
        uffici_1h_nmae  = compute_nmae_by_hour(uffici_1h_forecast,  uffici_ground_truth)
        uffici_12h_nmae = compute_nmae_by_hour(uffici_12h_forecast, uffici_ground_truth)
        # RMSE Irraggiamento
        irr_1h_rmse  = compute_rmse_by_hour(irr_1h_forecast,  irr_ground_truth)
        irr_12h_rmse = compute_rmse_by_hour(irr_12h_forecast, irr_ground_truth)

        # --- Plot di confronto ---
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(24), uffici_1h_nmae,  marker='o', label='Uffici 1h')
        plt.plot(np.arange(24), uffici_12h_nmae, marker='o', label='Uffici 12h', color='orange')
        plt.xlabel("Ora del giorno (0-23)")
        plt.ylabel("NMAE")
        plt.title("Confronto NMAE - Uffici 1h vs 12h")
        plt.legend()
        plt.grid(True)
        plt.savefig("TimeSeries/plots/confronto_nmae_uffici.png", dpi=300, bbox_inches='tight')
        if args.forecastplot:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(np.arange(24), irr_1h_rmse,  marker='o', label='Irraggiamento 1h')
        plt.plot(np.arange(24), irr_12h_rmse, marker='o', label='Irraggiamento 12h', color='orange')
        plt.xlabel("Ora del giorno (0-23)")
        plt.ylabel("RMSE")
        plt.title("Confronto RMSE - Irraggiamento 1h vs 12h")
        plt.legend()
        plt.grid(True)
        plt.savefig("TimeSeries/plots/confronto_rmse_irraggiamento.png", dpi=300, bbox_inches='tight')
        if args.forecastplot:
            plt.show()
        else:
            plt.close()

        # --- Ora destandardizziamo i forecast PRIMA di salvarli ---
        # (usiamo i parametri di standardizzazione per la colonna Potenza Uffici e Irraggiamento)
        uff_mean = TRAIN_MEAN["Potenza Uffici [W]"]
        uff_std  = TRAIN_STD["Potenza Uffici [W]"]
        irr_mean = TRAIN_MEAN["Irraggiamento [kWh/m2]"]
        irr_std  = TRAIN_STD["Irraggiamento [kWh/m2]"]

        # Creiamo i DataFrame con i forecast e destandardizziamo la colonna 'Forecast'
        timestamps = pd.date_range(start="01/07/2022 00:00:00", periods=forecast_steps, freq="H")

        df_uffici_1h = pd.DataFrame({"Timestamp": timestamps, "Forecast": uffici_1h_forecast})
        df_uffici_12h = pd.DataFrame({"Timestamp": timestamps, "Forecast": uffici_12h_forecast})
        df_irraggiamento_1h = pd.DataFrame({"Timestamp": timestamps, "Forecast": irr_1h_forecast})
        df_irraggiamento_12h = pd.DataFrame({"Timestamp": timestamps, "Forecast": irr_12h_forecast})

        # Destandardizzo la colonna Forecast (solo per i valori del target corrispondente)
        df_uffici_1h["Forecast"] = df_uffici_1h["Forecast"] * uff_std + uff_mean
        df_uffici_12h["Forecast"] = df_uffici_12h["Forecast"] * uff_std + uff_mean
        df_irraggiamento_1h["Forecast"] = df_irraggiamento_1h["Forecast"] * irr_std + irr_mean
        df_irraggiamento_12h["Forecast"] = df_irraggiamento_12h["Forecast"] * irr_std + irr_mean

        # Salviamo i file Excel in scala reale
        if not os.path.exists('TimeSeries/forecasted_data'):
            os.makedirs('TimeSeries/forecasted_data')
        df_uffici_1h.to_excel("TimeSeries/forecasted_data/uffici_1h_forecast.xlsx", index=False)
        df_uffici_12h.to_excel("TimeSeries/forecasted_data/uffici_12h_forecast.xlsx", index=False)
        df_irraggiamento_1h.to_excel("TimeSeries/forecasted_data/irraggiamento_1h_forecast.xlsx", index=False)
        df_irraggiamento_12h.to_excel("TimeSeries/forecasted_data/irraggiamento_12h_forecast.xlsx", index=False)

        print("\nForecasting completed and data saved (in real scale).")

    print("Process completed successfully.")

if __name__ == "__main__":
    main()