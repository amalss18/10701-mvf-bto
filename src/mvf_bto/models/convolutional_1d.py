import tensorflow as tf


class Convolutional1D(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        n_outputs,
        conv_filters_1=64,
        conv_kernel_size_1=8,
        pool_size=2,
        dense_units=128,
    ):
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv1D(
            filters=conv_filters_1,
            kernel_size=conv_kernel_size_1,
            activation="relu",
            input_shape=input_shape,
        )

        self.pooling_1 = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=dense_units, activation="leaky_relu")
        self.dense_2 = tf.keras.layers.Dense(units=50, activation="tanh")
        self.output_layer = tf.keras.layers.Dense(units=n_outputs)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.pooling_1(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense_2(x)
        return self.output_layer(x)
