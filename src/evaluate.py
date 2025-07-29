import tensorflow as tf

def evaluate_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    X_test = tf.random.normal((8, 100, 40))
    y_test = tf.random.uniform((8,), maxval=30, dtype=tf.int32)
    preds = model.predict(X_test)
    print("Predictions shape:", preds.shape)

if __name__ == "__main__":
    evaluate_model("models/lstm_attention.h5")
