# src/utils/model_utils.py

import tensorflow as tf
import numpy as np
import keras
from keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Lambda,
    GlobalAveragePooling1D,
    LayerNormalization
)
from keras.models import Model
from keras.optimizers import Nadam
from sklearn.utils.class_weight import compute_class_weight
import gensim.downloader as api
from keras.layers import TextVectorization, Embedding
from keras.optimizers import RMSprop

    
from keras.layers import Layer

class AbsoluteDifference(Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.abs(x1 - x2)
    
# Text Vectorizer
def build_text_vectorizer(X_train, max_tokens=25_000, seq_len=30):
    from keras.layers import TextVectorization

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len
    )

    q1 = [tokens_to_text(x) for x in X_train[:, 0]]
    q2 = [tokens_to_text(x) for x in X_train[:, 1]]

    vectorizer.adapt(q1 + q2)
    return vectorizer


# GloVe Embedding Layer
def build_glove_embedding_layer(vectorizer):
    glove_model = api.load("glove-twitter-200")
    embedding_dim = glove_model.vector_size

    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for idx, word in enumerate(vocab):
        if word in glove_model:
            embedding_matrix[idx] = glove_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=True,
        mask_zero=False
    )

    return embedding_layer


# Siamese Model Architecture
def build_siamese_model(vectorizer, embedding_layer, max_len=30):
    # Inputs

    q1_input = Input(shape=(), dtype="string", name="q1")
    q2_input = Input(shape=(), dtype="string", name="q2")

    q1_int = vectorizer(q1_input)
    q2_int = vectorizer(q2_input)

    encoder_input = Input(shape=(None,), dtype="int32")

    x = embedding_layer(encoder_input)
    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization()(x)

    encoder = Model(encoder_input, x, name="shared_encoder")

    q1_vec = encoder(q1_int)
    q2_vec = encoder(q2_int)

    text_features = keras.ops.abs(q1_vec - q2_vec)

    x = Dense(64, activation="relu")(text_features)
    x = Dropout(0.4)(x)

    output = Dense(1, activation="sigmoid", name="similarity")(x)

    model = Model(
        inputs=[q1_input, q2_input],
        outputs=output
    )

    return model


# Class Weights
def compute_class_weights(y):
    """
    Computes class weights for binary classification.
    Ensures y is 1D.
    """
    y = np.asarray(y).ravel()   # 🔑 FIX: flatten (N, 1) → (N,)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )

    return {i: weights[i] for i in range(len(weights))}

# Compile Model
def compile_model(model):
    model.compile(
        optimizer=Nadam(),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model

def tokens_to_text(x):
    """
    Converts token lists or strings into a clean string.
    """
    if isinstance(x, list):
        return " ".join(x)
    elif isinstance(x, str):
        return x
    else:
        return str(x)

