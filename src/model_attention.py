import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_lstm_attention(input_dim, output_dim):
    inputs = tf.keras.Input(shape=(None, input_dim))
    lstm_out = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    query = tf.keras.layers.LSTM(256)(lstm_out)
    context_vector, attention_weights = BahdanauAttention(128)(query, lstm_out)
    output = tf.keras.layers.Dense(output_dim, activation='softmax')(context_vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
