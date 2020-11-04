from tensorflow import keras


class NeuralNetwork:
    def __init__(self, input_dim,
                 input_neurons, optimizer_lr=0.001, output_dim=2):
        self.input_dim = input_dim
        self.input_neurons = input_neurons
        self.hidden_neurons = 24
        self.optimizer_lr = optimizer_lr
        self.output_dim = output_dim
        self.keras_model = self.build_model()

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
        print(model.summary())
        return model


def build_lstm():
    """
    WIP
    :return:
    """
    lstm_autoencoder = keras.Sequential()
    # Encoder
    lstm_autoencoder.add(
        keras.layers.LSTM(32, activation='relu', input_shape=(),
                          return_sequences=True))
    lstm_autoencoder.add(keras.layers.LSTM(16, activation='relu', return_sequences=False))
    lstm_autoencoder.add(keras.layers.RepeatVector(256))
    # Decoder
    lstm_autoencoder.add(keras.layers.LSTM(16, activation='relu', return_sequences=True))
    lstm_autoencoder.add(keras.layers.LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder.add(keras.layers.TimeDistributed(keras.layers.Dense(2)))
    lstm_autoencoder.compile(loss='mse',  # Loss function: Mean Squared Error
                             optimizer=keras.optimizers.Adam(
                                 lr=0.001))  # Optimaizer: Adam (Feel free to check other options)

    lstm_autoencoder.summary()
    return lstm_autoencoder
