import tensorflow as tf


class BaselineLSTM(tf.keras.Model):
    def __init__(
            self,
            batch_input_shape,
            n_outputs,
            lstm1_units=32,
            lstm2_units=16,
            dense1_units=16,
            dense2_units=8,
    ):
        super().__init__()
        self.lstm_1 = tf.keras.layers.LSTM(
            units=lstm1_units,
            return_sequences=True,
            stateful=True,
            batch_input_shape=batch_input_shape,
        )
        self.lstm_2 = tf.keras.layers.LSTM(
            units=lstm2_units,
            return_sequences=False,
            stateful=False,
        )
        self.dense_1 = tf.keras.layers.Dense(units=dense1_units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(units=dense2_units)
        self.output_layer = tf.keras.layers.Dense(units=n_outputs)

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.output_layer(x)
