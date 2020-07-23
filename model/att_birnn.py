from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from model.attention import Attention

def create_model(vocab_size, tag_size, max_len, rnn_units, embedding_dims, emb_matrix=None):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dims, input_length=max_len) if emb_matrix is None else Embedding(input_dim=vocab_size, output_dim=embedding_dims, weights=[emb_matrix], input_length=max_len, trainable=False)
    bi_rnn = Bidirectional(LSTM(units=rnn_units, return_sequences=True))
    attention = Attention(max_len)
    classifier = Dense(1, activation="sigmoid")
    
    embedding = embedding_layer(inputs)
    x = bi_rnn(embedding)
    x = attention(x)
    output = classifier(x)

    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model
