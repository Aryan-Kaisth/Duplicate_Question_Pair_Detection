# src/components/model_trainer.py

import os, sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.utils.model_utils import (
    build_text_vectorizer,
    build_glove_embedding_layer,
    build_siamese_model,
    compute_class_weights,
    compile_model,
    tokens_to_text,

)


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join(
        "artifacts", "model_trainer", "siamese.keras"
    )

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
        logging.info("ModelTrainer initialized")

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training started")

            # ---------------- Vectorizer ----------------
            vectorizer = build_text_vectorizer(X_train)

            # ---------------- Embedding ----------------
            embedding_layer = build_glove_embedding_layer(vectorizer)

            # ---------------- Model ----------------
            model = build_siamese_model(vectorizer, embedding_layer)
            model = compile_model(model)

            logging.info(model.summary(show_trainable=True, line_length=115))

            # ---------------- Data ----------------
            import tensorflow as tf

            q1_train = tf.constant(
                [tokens_to_text(x) for x in X_train[:, 0]],
                dtype=tf.string
            )
            q2_train = tf.constant(
                [tokens_to_text(x) for x in X_train[:, 1]],
                dtype=tf.string
            )

            q1_test = tf.constant(
                [tokens_to_text(x) for x in X_test[:, 0]],
                dtype=tf.string
            )
            q2_test = tf.constant(
                [tokens_to_text(x) for x in X_test[:, 1]],
                dtype=tf.string
            )

            y_train = tf.constant(y_train.ravel(), dtype=tf.int32)
            y_test = tf.constant(y_test.ravel(), dtype=tf.int32)

            class_weight = compute_class_weights(y_train)

            # ---------------- Training ----------------
            model.fit(
                {"q1": q1_train, "q2": q2_train},
                y_train,
                batch_size=256,
                epochs=5,
                validation_split=0.3,
                class_weight=class_weight
            )

            # ---------------- Evaluation ----------------
            metrics = model.evaluate(
                {"q1": q1_test, "q2": q2_test},
                y_test
            )
            logging.info(f"Test metrics: {metrics}")

            # ---------------- Save ----------------
            model.save(self.config.model_file_path)
            logging.info(f"Model saved at {self.config.model_file_path}")

            return model

        except Exception as e:
            logging.error("Error during model training")
            raise CustomException(e, sys)


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation

    # Paths to train and test data
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize the transformer
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    model = ModelTrainer()
    model.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)