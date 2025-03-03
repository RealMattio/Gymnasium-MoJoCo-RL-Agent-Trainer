import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, data_path:str):
        """
        Initialize the preprocessor.

        :param data_path: (str) The path to the dataset wich contain the data to train.
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        """
        Load the data from the dataset.

        :return: (pd.DataFrame) The dataset.
        """
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xls') or self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xls/.xlsx file.")
        return self.data
    
    def preprocess_data(self, columns:list[str]=['Potenza Uffici [W]','Temperatura [K]','Nuvolosità [%]','Irraggiamento [kWh/m2]','Day_sin','Day_cos']):
        """
        Preprocess the data.
        :param columns: (list) The columns to keep in the preprocessed data. These columns will be used to train the model.
        :return: (pd.DataFrame) The preprocessed data.
        """
        # 1. Conversione della colonna Data in formato datetime
        self.data['Data'] = pd.to_datetime(self.data['Data'], format='%d.%m.%Y %H:%M:%S')

        # 2. Estrazione dell'ora dalla colonna Ora (ad esempio "14:00" -> 14)
        self.data['Ora'] = self.data['Ora'].str.split(':').str[0].astype(int)

        # 3. Calcolo delle trasformazioni orarie
        self.data['Day_sin'] = np.sin(2 * np.pi * self.data['Ora'] / 24)
        self.data['Day_cos'] = np.cos(2 * np.pi * self.data['Ora'] / 24)

        # 4. Costruzione della colonna date_time combinando la data (senza orario) e l'ora
        self.data['date_time'] = pd.to_datetime(
            self.data['Data'].dt.strftime('%Y-%m-%d') + ' ' + self.data['Ora'].astype(str).str.zfill(2) + ':00:00',
            format='%Y-%m-%d %H:%M:%S'
        )
        
        # 5. Rimozione delle colonne non necessarie
        self.data = self.data[columns]
        return self.data
    
    def divide_train_val_test_data(self, standardize:bool=True):
        """
        Divide the data into training, validation and test sets.
        :param standardize: (bool) If True, the data will be standardized. This should be always True.
        :return: (pd.DataFrame) The standardized data.
        """
        # 1. Suddivisione e standardizzazione dei dati
        mean_and_std = None
        n = len(self.data)
        self.train_data = self.data.iloc[:int(n*0.7)]
        self.val_data   = self.data.iloc[int(n*0.7):int(n*0.9)]
        self.test_data  = self.data.iloc[int(n*0.9):]
        if standardize:
            self.standardize_data()            
        return self.train_data, self.val_data, self.test_data
    
    def standardize_data(self, train_mean=None, train_std=None):
        """
        Standardize the data. This method can be used in two ways: if is used on the train data, the mean and the standard deviation will be computed and saved. 
        If is used on the data used for forecasting mean and standard deviation must be passed as arguments.
        :param train_mean: (float) The mean of the training data if available.
        :param train_std: (float) The standard deviation of the training data if available.
        """
        if train_mean is not None and train_std is not None:
            self.data = (self.data - train_mean) / train_std
            return None
        else:
            self.train_mean = self.train_data.mean()
            self.train_std  = self.train_data.std()


            self.train_data = (self.train_data - self.train_mean) / self.train_std
            self.val_data   = (self.val_data - self.train_mean) / self.train_std
            self.test_data  = (self.test_data - self.train_mean) / self.train_std
            return (self.train_mean, self.train_std)