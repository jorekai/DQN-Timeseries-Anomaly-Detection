from tensorflow import keras
from environment import WindowStateFunctions


def build_model():
    model = keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
    model.add(
        keras.layers.Dense(WindowStateFunctions.SLIDE_WINDOW_SIZE + 1, input_dim=WindowStateFunctions.SLIDE_WINDOW_SIZE,
                           activation='relu'))  # [Input] -> Layer 1
    model.add(keras.layers.Dense(WindowStateFunctions.SLIDE_WINDOW_SIZE * 4, activation='relu'))  # Layer 3 -> [output]
    model.add(keras.layers.Dense(WindowStateFunctions.SLIDE_WINDOW_SIZE * 4, activation='relu'))  # Layer 3 -> [output]
    model.add(keras.layers.Dense(2, activation='linear'))  # Layer 3 -> [output]
    model.compile(loss='mse',  # Loss function: Mean Squared Error
                  optimizer=keras.optimizers.Adam(
                      lr=0.001))  # Optimaizer: Adam (Feel free to check other options)
    return model


def build_lstm():
    """
    WIP
    :return:
    """
    lstm_autoencoder = keras.Sequential()
    # Encoder
    lstm_autoencoder.add(
        keras.layers.LSTM(32, activation='relu', input_shape=(None, WindowStateFunctions.SLIDE_WINDOW_SIZE),
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
