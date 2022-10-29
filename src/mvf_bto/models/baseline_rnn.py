import tensorflow as tf


class BaselineRNN(tf.keras.Model):
    def __init__(
            self,
            batch_input_shape,
            n_outputs,
            rnn1_units=32,
            rnn2_units=16,
            dense1_units=16,
            dense2_units=8,
    ):
        super().__init__()
        self.rnn_1 = tf.keras.layers.SimpleRNN(
            units=rnn1_units,
            return_sequences=True,
            stateful=True,
            batch_input_shape=batch_input_shape,
        )
        self.rnn_1 = tf.keras.layers.SimpleRNN(
            units=rnn2_units,
            return_sequences=False,
            stateful=False,
        )
        self.dense_1 = tf.keras.layers.Dense(units=dense1_units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(units=dense2_units)
        self.output_layer = tf.keras.layers.Dense(units=n_outputs)

    def call(self, inputs):
        x = self.rnn_1(inputs)
        x = self.rnn_1(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.output_layer(x)