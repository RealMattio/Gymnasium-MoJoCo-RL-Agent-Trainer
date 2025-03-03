import numpy as np
import tensorflow as tf

class DatasetModeller:
    @staticmethod
    def make_dataset(data, input_width, label_width, shift, batch_size, target_col="Potenza Uffici [W]"):
        # Converte il DataFrame in array numpy
        data_array = np.array(data, dtype=np.float32)
        total_window_size = input_width + shift
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data_array,
            targets=None,
            sequence_length=total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=batch_size
        )
        # Ottieni l'indice della colonna target
        target_col_index = data.columns.get_loc(target_col)
        def split_window(window):
            inputs = window[:, :input_width, :]
            # Seleziona solo la colonna target per le etichette:
            labels = window[:, input_width:input_width+label_width, target_col_index]
            # labels avrà forma (batch, label_width)
            return inputs, labels
        
        ds = ds.map(split_window)
        return ds

    @staticmethod
    def create_sequences_df(df, input_width=24, out_steps=24, target_col="Potenza Uffici [W]"):
        '''
        Crea sequenze di input e output per un DataFrame.
        :param df: (pd.DataFrame) Il DataFrame contenente i dati.
        :param input_width: (int) La larghezza della sequenza di input.
        :param out_steps: (int) Il numero di passi temporali previsti.
        :param target_col: (str) Il nome della colonna target.
        :return: (np.array, np.array) Le sequenze di input e le etichette.
        '''
        sequences = []
        labels = []
        for i in range(len(df) - input_width - out_steps + 1):
            seq_input = df.iloc[i : i + input_width].values
            # Selezioniamo solo la colonna target
            seq_label = df.iloc[i + input_width : i + input_width + out_steps][target_col].values
            sequences.append(seq_input)
            labels.append(seq_label)
        sequences = np.array(sequences)
        labels = np.array(labels)
        # Per out_steps > 1 manteniamo la shape (N, out_steps)
        return sequences, labels
    
    @staticmethod
    def extract_labels(dataset):
        # Initialize an empty list to collect labels from all batches
        all_labels = []
        
        # Iterate through each batch in the dataset
        for inputs, labels in dataset:
            # Convert TensorFlow tensor to numpy array and append to list
            all_labels.append(labels.numpy())
        
        # Combine all batches into a single numpy array
        all_labels = np.concatenate(all_labels, axis=0)
        return all_labels
    