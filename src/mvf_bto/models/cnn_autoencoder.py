import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def encoder(input_shape, latent_dimension: int = 32, model_name: str = "encoder"):
    """
    Parameters
    ----------
    latent_dimensions
        Number of dimensions in the latent space.
    model_name
        Name of the encoder (passed to model constructor).

    Returns
    -------
    A 1D convolutional encoder.
    """

    # convolutional layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=5, padding="same", strides=1, activation="relu")(inputs)
    x = layers.Conv1D(filters=16, kernel_size=3, padding="same", strides=1, activation="relu")(x)

    # dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(8 * 16, activation="relu", name="dense1")(x)
    # x = layers.Dense(8, activation="relu", name="dense2")(x)
    z = layers.Dense(latent_dimension, activation="relu", name="dense3")(x)
    z = layers.Reshape(target_shape=(32, 1), name="reshape")(z)
    return keras.Model(inputs, z, name=model_name)


def decoder(latent_dimension: int = 32, model_name: str = "decoder"):
    """
    Parameters
    ----------
    latent_dimensions
        Number of dimensions in the latent space.
    model_name
        Name of the decoder (passed to model constructor).

    Returns
    -------
    A 1D convolutional decoder.
    """

    # dense layers
    inputs = keras.Input(shape=(latent_dimension,))
    x = layers.Dense(8, activation="relu", name="decode_dense2")(inputs)
    x = layers.Dense(7 * 16, activation="relu", name="decode_dense1")(x)

    # convolutional layers
    x = layers.Reshape(target_shape=(7, 16))(x)
    x = layers.Conv1DTranspose(filters=16, kernel_size=3, padding="same", strides=1, activation="relu")(x)
    x = layers.Conv1DTranspose(filters=32, kernel_size=5, padding="same", strides=1, activation="relu")(x)
    outputs = layers.Conv1DTranspose(filters=4, kernel_size=5, padding="same")(x)

    return keras.Model(inputs, outputs, name=model_name)


def forecaster(latent_dimension: int = 32, model_name="forecaster"):
    inputs = keras.Input(batch_shape=(10, 32, 1))
    x = layers.LSTM(32, name="LSTM", stateful=True,
                    )(inputs)
    outputs = layers.Dense(latent_dimension, activation="leaky_relu", name="latent_output")(x)
    return keras.Model(inputs, outputs, name=model_name)


class Autoencoder(keras.Model):

    #     def __init__(self, encoder, decoder, forecaster, alpha=1, **kwargs):
    def __init__(self, encoder, decoder, forecaster, alpha=1, **kwargs):
        super().__init__()

        # loss hyperparameter
        self.alpha = alpha

        # loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.forecasting_loss_tracker = keras.metrics.Mean(name="forecast_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        # layers
        self.encoder = encoder
        self.decoder = decoder
        self.forecaster = forecaster

    @property
    def metrics(self):
        return [
            self.forecasting_loss_tracker,
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def call(self, inputs):
        """Returns the latent representation and the reconstruction."""
        z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        latent_forecast = self.forecaster(z)
        forecast_reconstruction = self.decoder(latent_forecast)

        return z, reconstruction, latent_forecast, forecast_reconstruction

    def compute_loss(self, X, y, **args):
        """Optimizes reconstruction loss and forecast loss."""
        # executes forward pass
        print(X)
        z, reconstruction, latent_forecast, forecast_reconstruction = self(X)

        # computes total loss (reconstruction and forecast)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(X, reconstruction)))
        prediction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(y, forecast_reconstruction)))
        total_loss = reconstruction_loss + self.alpha * prediction_loss

        # updates metrics and returns loss
        self.total_loss_tracker.update_state(total_loss)
        self.forecasting_loss_tracker.update_state(prediction_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return total_loss

    def train_step(self, inputs, **args):
        X, y = inputs
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(X, y)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "forecasting_loss": self.forecasting_loss_tracker.result(),
        }