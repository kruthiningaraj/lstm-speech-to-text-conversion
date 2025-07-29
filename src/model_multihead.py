import tensorflow as tf

def build_lstm_multihead_attention(input_dim, output_dim, num_heads=4):
    inputs = tf.keras.Input(shape=(None, input_dim))
    lstm_out = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(lstm_out, lstm_out)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
    output = tf.keras.layers.Dense(output_dim, activation='softmax')(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
