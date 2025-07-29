import tensorflow as tf
import numpy as np

def infer(model_path, audio_features):
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(np.expand_dims(audio_features, axis=0))
    decoded = np.argmax(preds, axis=-1)
    return decoded

if __name__ == "__main__":
    dummy_input = np.random.rand(100, 40)
    print("Decoded output:", infer("models/lstm_attention.h5", dummy_input))
