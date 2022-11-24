import tensorflow as tf


class BaselineLSTM(tf.keras.Model):
    def __init__(
            self,
            batch_input_shape,
            n_outputs,
            nf_steps=1,
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
        self.dense_3 = tf.keras.layers.Dense(units=nf_steps*n_outputs)
        self.output_layer = tf.keras.layers.Reshape([nf_steps, n_outputs])

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.output_layer(x)


class LSTMAutoregression(tf.keras.Model):
    def __init__(
            self,
            n_outputs,
            nf_steps=1,
            lstm1_units=32,
            dense1_units=32,
            dense2_units=8,
            out_steps = 5
    ):
        super().__init__()
        self.out_steps = out_steps
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm1_units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

        self.dense_1 = tf.keras.layers.Dense(units=dense1_units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(units=dense2_units)
        self.dense_3 = tf.keras.layers.Dense(units=nf_steps*n_outputs)
        # self.output_layer = tf.keras.layers.Reshape([nf_steps, n_outputs])
    
    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)

        x = self.dense_1(x)
        x = self.dense_2(x)
        prediction = self.dense_3(x)
        # prediction = self.output_layer(prediction)

        return prediction, state        

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        # print(prediction)
        # print(state)
        for n in range(1, self.out_steps):
            x = prediction

            x, state = self.lstm_cell(x, states=state, training=training)
            x = self.dense_1(x)
            x = self.dense_2(x)
            prediction = self.dense_3(x)
            # prediction = self.output_layer(prediction)

            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions

