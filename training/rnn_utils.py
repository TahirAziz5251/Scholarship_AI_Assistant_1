from tensorflow.keras.models import load_model

def save_rnn_model(model, path="rnn_model.h5"):
    model.save(path)

def load_rnn_model(path="rnn_model.h5"):
    return load_model(path)