import tensorflow as tf
from preprocess import load_dataset
from model_base import build_lstm_model
from model_attention import build_lstm_attention
from model_multihead import build_lstm_multihead_attention
from loss_functions import cross_entropy_loss
import os

def train_model(model_type='base'):
    input_dim, output_dim = 40, 30  # example dims
    if model_type == 'base':
        model = build_lstm_model(input_dim, output_dim)
    elif model_type == 'attention':
        model = build_lstm_attention(input_dim, output_dim)
    else:
        model = build_lstm_multihead_attention(input_dim, output_dim)

    model.compile(optimizer='adam', loss=cross_entropy_loss, metrics=['accuracy'])

    # Dummy placeholders for actual dataset
    X_train, y_train = tf.random.normal((32, 100, input_dim)), tf.random.uniform((32,), maxval=output_dim, dtype=tf.int32)
    X_val, y_val = tf.random.normal((8, 100, input_dim)), tf.random.uniform((8,), maxval=output_dim, dtype=tf.int32)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_{model_type}.h5")

if __name__ == "__main__":
    train_model('attention')
