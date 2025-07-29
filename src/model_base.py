import tensorflow as tf

def build_lstm_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(None, input_dim)),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model
