from tensorflow import keras

from environment import BatchLearning


def build_model():
    model = keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
    model.add(keras.layers.Dense(24, input_dim=BatchLearning.SLIDE_WINDOW_SIZE,
                                 activation='relu'))  # [Input] -> Layer 1
    model.add(keras.layers.Dense(48, activation='relu'))  # Layer 3 -> [output]
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(48, activation='relu'))  # Layer 3 -> [output]
    model.add(keras.layers.Dense(2, activation='linear'))  # Layer 3 -> [output]
    model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                  optimizer=keras.optimizers.Adam(
                      lr=0.001))  # Optimaizer: Adam (Feel free to check other options)
    return model
