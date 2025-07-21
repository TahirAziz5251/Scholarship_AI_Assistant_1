from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding


def build_run_model(vocab_size=10000, embedding_dim=128, lstm_units=256, input_length=100):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
