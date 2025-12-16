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
    LayerNormalization,
    Concatenate
)
from keras.models import Model
from keras.optimizers import Nadam
from sklearn.utils.class_weight import compute_class_weight
import gensim.downloader as api
from keras.layers import TextVectorization, Embedding
from keras.optimizers import RMSprop
from keras.layers import Layer
    
# Text Vectorizer
def build_text_vectorizer(q1, q2, max_tokens=40_000, seq_len=30):
    from keras.layers import TextVectorization

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len
    )
    vectorizer.adapt(
        np.concatenate([q1, q2])
    )
    return vectorizer

# fasttext Embedding Layer
def build_fasttext_embedding_layer(vectorizer, trainable=True):
    fasttext_model = api.load("fasttext-wiki-news-subwords-300")
    embedding_dim = fasttext_model.vector_size

    vocab = vectorizer.get_vocabulary()
    # Embedding dimension
    vocab_size = len(vocab)

    # Initialize embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for idx, word in enumerate(vocab):
        if word in fasttext_model:
            embedding_matrix[idx] = fasttext_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(
                scale=0.6, size=(embedding_dim,)
            )

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=trainable,
        mask_zero=False
    )

    return embedding_layer

# Siamese Model Architecture
def build_siamese_model(
    vectorizer,
    embedding_layer,
    num_engineered_features: int,
    dropout_rate: float = 0.8
) -> Model:
    
    q1_input = Input(shape=(), dtype="string", name="q1")
    q2_input = Input(shape=(), dtype="string", name="q2")

    feat_input = Input(
        shape=(num_engineered_features,),
        dtype="float32",
        name="engineered_features"
    )

    q1_int = vectorizer(q1_input)
    q2_int = vectorizer(q2_input)

    encoder_input = Input(shape=(None,), dtype="int32")

    x = embedding_layer(encoder_input)
    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization()(x)

    encoder = Model(encoder_input, x, name="siamese_encoder")

    q1_vec = encoder(q1_int)
    q2_vec = encoder(q2_int)

    text_diff = tf.keras.ops.abs(q1_vec - q2_vec)

    f = Dense(32, activation="relu")(feat_input)

    combined = Concatenate()([text_diff, f])

    x = Dense(32, activation="relu")(combined)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(1, activation="sigmoid", name="similarity")(x)

    model = Model(
        inputs=[q1_input, q2_input, feat_input],
        outputs=output
    )

    model.compile(
        optimizer=RMSprop(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc")]
    )

    return model

# Class Weights
def compute_class_weights(y):
    """
    Computes class weights for binary classification.
    Ensures y is 1D.
    """
    y = np.asarray(y).ravel()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    return {i: weights[i] for i in range(len(weights))}

