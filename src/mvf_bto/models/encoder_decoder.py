import tensorflow as tf


class EncoderDecoder(tf.keras.Model):
    def __init__(
        self,
        batch_input_shape,
        n_outputs,
        nf_steps=1,
        lstm_units=100,
    ):
        super().__init__()
        self.lstm_encoder = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            activation="relu",
            batch_input_shape=batch_input_shape,
        )

        self.dilate = tf.keras.layers.RepeatVector(n_outputs)
        self.lstm_decoder = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            activation="relu",
        )
        self.time_distributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=nf_steps))

    def call(self, inputs):
        x = self.lstm_encoder(inputs)
        x = self.dilate(x)
        x = self.lstm_decoder(x)
        return self.time_distributed(x)
