from tensorflow import keras

from agents import DQNWAgent
from environment import BatchLearning


def build_model():
    model = keras.Sequential()  # linear stack of layers
    model.add(keras.layers.Dense(BatchLearning.SLIDE_WINDOW_SIZE + 1, input_dim=BatchLearning.SLIDE_WINDOW_SIZE,
                                 activation='relu'))  # [Input] -> Layer 1
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(loss='mse',  # Loss function: Mean Squared Error
                  optimizer=keras.optimizers.Adam(
                      lr=0.001))  # Optimizer: Adam
    return model


def build_lstm():
    lstm_autoencoder = keras.Sequential()
    # Encoder
    lstm_autoencoder.add(
        keras.layers.LSTM(BatchLearning.SLIDE_WINDOW_SIZE + 1, activation='tanh',
                          input_shape=(1, BatchLearning.SLIDE_WINDOW_SIZE),
                          return_sequences=True))
    # lstm_autoencoder.add(keras.layers.Dense(128, activation='relu'))
    lstm_autoencoder.add(keras.layers.LSTM(256, activation='tanh', return_sequences=True))
    # lstm_autoencoder.add(keras.layers.RepeatVector(256))
    # # # Decoder
    # lstm_autoencoder.add(keras.layers.LSTM(16, activation='relu', return_sequences=True))
    # lstm_autoencoder.add(keras.layers.LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder.add(keras.layers.Flatten())
    lstm_autoencoder.add(keras.layers.Dense(2, activation='linear'))
    lstm_autoencoder.compile(loss='mse',  # Loss function: Mean Squared Error
                             optimizer=keras.optimizers.Adam(
                                 lr=0.001))  # Optimaizer: Adam (Feel free to check other options)

    lstm_autoencoder.summary()
    return lstm_autoencoder
