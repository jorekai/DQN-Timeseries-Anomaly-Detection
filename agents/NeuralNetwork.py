from tensorflow import keras


class NeuralNetwork:
    def __init__(self, input_dim,
                 input_neurons, optimizer_lr=0.00001, output_dim=2, hidden_neurons=24, type="standard"):
        self.input_dim = input_dim
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.optimizer_lr = optimizer_lr
        self.output_dim = output_dim
        if type == "standard":
            self.keras_model = self.build_model()
        elif type == "lstm":
            self.keras_model = self.build_lstm()
        elif type == "lstm_binary":
            self.keras_model = self.build_lstm(False)
        elif type == "lstm_cnn":
            self.keras_model = self.build_lstm_cnn()

    def build_model(self):
        model = keras.Sequential()  # https://keras.io/models/sequential/
        model.add(
            keras.layers.Dense(self.hidden_neurons, input_dim=self.input_dim,
                               activation='relu'))  # [Input] -> Layer 1
        model.add(
            keras.layers.Dense(self.hidden_neurons, activation='relu'))  # Layer 2 -> [hidden1]
        model.add(
            keras.layers.Dense(self.hidden_neurons, activation='relu'))  # Layer 3 -> [hidden2]
        model.add(keras.layers.Dense(self.output_dim, activation='linear'))  # Layer 4 -> [out]
        model.compile(loss='mse',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(
                          lr=self.optimizer_lr))  # Optimizer: Adam (Feel free to check other options)
        return model

    def build_lstm(self, slide_window=True):
        lstm_autoencoder = keras.Sequential()
        # Encoder
        if slide_window:
            lstm_autoencoder.add(
                keras.layers.LSTM(self.input_dim, activation='tanh',
                                  batch_input_shape=(None, 1, self.input_dim),
                                  return_sequences=True))
        else:
            lstm_autoencoder.add(
                keras.layers.LSTM(self.input_dim, activation='tanh',
                                  input_shape=([self.input_dim, 2]),
                                  return_sequences=True))
        lstm_autoencoder.add(keras.layers.LSTM(self.hidden_neurons, activation='tanh', return_sequences=True))
        lstm_autoencoder.add(keras.layers.Dense(self.hidden_neurons, activation='relu'))
        lstm_autoencoder.add(keras.layers.Flatten())
        lstm_autoencoder.add(keras.layers.Dense(2, activation='linear'))
        lstm_autoencoder.compile(loss='mse',  # Loss function: Mean Squared Error
                                 optimizer=keras.optimizers.Adam(
                                     lr=self.optimizer_lr))  # Optimaizer: Adam (Feel free to check other options)

        lstm_autoencoder.summary()
        return lstm_autoencoder

    def build_lstm_cnn(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu',
                                      input_shape=(self.input_dim, 2)))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.LSTM(self.hidden_neurons, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dense(self.hidden_neurons, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mse',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(
                          lr=self.optimizer_lr))  # Optimaizer: Adam (Feel free to check other options)

        model.summary()
        return model
