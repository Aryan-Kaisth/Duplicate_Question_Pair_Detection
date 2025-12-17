# src/utils/model_utils.py

import tensorflow as tf
import numpy as np
import keras
from keras.layers import (
    Input,
    Conv1D,
    Dense,
    Dropout,
    Lambda,
    GlobalMaxPooling1D,
    LayerNormalization,
    Concatenate,
    Subtract, 
    Multiply, 
    Dot,
    GlobalAveragePooling1D
)
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.utils.class_weight import compute_class_weight
import gensim.downloader as api
from keras.layers import TextVectorization, Embedding
from keras.regularizers import l2
# Text Vectorizer

def build_text_vectorizer(q1, q2, max_tokens=25_000, seq_len=30):
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
def build_fasttext_embedding_layer(vectorizer):
    fasttext_model = api.load('glove-wiki-gigaword-100')
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
        trainable=True,
        mask_zero=False
    )
    return embedding_layer

# Siamese Model Architecture
def build_siamese_cnn_model(vectorizer, embedding_layer):
    q1_input = Input(shape=(), dtype="string", name="q1")
    q2_input = Input(shape=(), dtype="string", name="q2")

    q1_int = vectorizer(q1_input)
    q2_int = vectorizer(q2_input)

    encoder_input = Input(shape=(None,), dtype="int32")

    x = embedding_layer(encoder_input)

    conv2 = Conv1D(32, 2, padding="same", activation="relu")(x)
    conv3 = Conv1D(32, 3, padding="same", activation="relu")(x)
    conv4 = Conv1D(32, 4, padding="same", activation="relu")(x)
    conv5 = Conv1D(32, 5, padding="same", activation="relu")(x)

    conv_out = Concatenate()([conv2, conv3, conv4, conv5])

    avg_pool = GlobalAveragePooling1D()(conv_out)
    max_pool = GlobalMaxPooling1D()(conv_out)

    text_vec = Concatenate()([avg_pool, max_pool])
    text_vec = LayerNormalization()(text_vec)

    encoder = Model(
        encoder_input,
        text_vec,
        name="siamese_encoder"
    )

    q1_vec = encoder(q1_int)
    q2_vec = encoder(q2_int)

    diff = keras.ops.abs(q1_vec - q2_vec)
    mul  = q1_vec * q2_vec
    cos  = Dot(axes=1, normalize=True)([q1_vec, q2_vec])

    combined = Concatenate()([diff, mul, cos])

    x = Dense(32, activation="relu")(combined)
    x = Dropout(0.4)(x)

    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation="sigmoid", name="similarity")(x)

    model = Model(
        inputs=[q1_input, q2_input],
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

