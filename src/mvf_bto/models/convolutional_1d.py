import tensorflow as tf


class Convolutional1D(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        n_outputs,
        conv_filters=64,
        conv_kernel_size=2,
        pool_size=2,
        dense_units=64,
    ):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            activation="relu",
            input_shape=input_shape,
        )

        self.pooling = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=dense_units)
        self.output_layer = tf.keras.layers.Dense(units=n_outputs)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)