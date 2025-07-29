import tensorflow as tf

def ctc_loss(y_true, y_pred):
    return tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=[len(y_true)]*len(y_true), logit_length=[y_pred.shape[1]]*len(y_true), logits_time_major=False, blank_index=-1)

def cross_entropy_loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
