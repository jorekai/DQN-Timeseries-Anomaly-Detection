from tensorflow import keras


class NeuralNetwork:
    def __init__(self, input_dim,
                 input_neurons, optimizer_lr=0.001, output_dim=2, hidden_neurons=24, type="standard"):
        self.input_dim = input_dim
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.optimizer_lr = optimizer_lr
        self.output_dim = output_dim
        if type == "standard":
            self.keras_model = self.build_model()
        elif type == "lstm":
            self.keras_model = self.build_lstm()

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

    def build_lstm(self):
        lstm_autoencoder = keras.Sequential()
        # Encoder
        lstm_autoencoder.add(
            keras.layers.LSTM(self.input_dim, activation='tanh',
                              batch_input_shape=(None, 1, self.input_dim),
                              return_sequences=True))
        lstm_autoencoder.add(keras.layers.LSTM(256, activation='tanh', return_sequences=True))
        # lstm_autoencoder.add(
        #    keras.layers.Dense(self.hidden_neurons, activation='relu'))  # Layer 2 -> [hidden1]
        lstm_autoencoder.add(keras.layers.Flatten())
        lstm_autoencoder.add(keras.layers.Dense(2, activation='linear'))
        lstm_autoencoder.compile(loss='mse',  # Loss function: Mean Squared Error
                                 optimizer=keras.optimizers.Adam(
                                     lr=0.001))  # Optimaizer: Adam (Feel free to check other options)

        lstm_autoencoder.summary()
        return lstm_autoencoder
