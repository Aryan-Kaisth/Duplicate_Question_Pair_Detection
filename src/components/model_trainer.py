import os, sys
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

from src.logger import logging
from src.exception import CustomException

from src.utils.model_utils import (
    build_text_vectorizer,
    build_fasttext_embedding_layer,
    build_siamese_model,
    compute_class_weights
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
        logging.info("[INIT] ModelTrainer initialized")

    def initiate_model_trainer(self, X_train_df, X_test_df, y_train, y_test):
        try:
            logging.info("[TRAINER] Model training started")

            # Split inputs from DataFrame
            q1_train = X_train_df["question1"].values
            q2_train = X_train_df["question2"].values

            q1_test = X_test_df["question1"].values
            q2_test = X_test_df["question2"].values

            feature_cols = [
                col for col in X_train_df.columns
                if col not in ["question1", "question2"]
            ]

            Xf_train = X_train_df[feature_cols].values
            Xf_test = X_test_df[feature_cols].values

            logging.info(
                f"[TRAINER] Text shapes: q1={q1_train.shape}, q2={q2_train.shape}"
            )
            logging.info(
                f"[TRAINER] Engineered feature shape: {Xf_train.shape}"
            )

            # Text Vectorizer (adapt on TRAIN text only)
            vectorizer = build_text_vectorizer(
                q1=q1_train,
                q2=q2_train
            )

            # Embedding Layer (FastText)
            embedding_layer = build_fasttext_embedding_layer(
                vectorizer=vectorizer,
                trainable=False
            )

            # Siamese Model
            model = build_siamese_model(
                vectorizer=vectorizer,
                embedding_layer=embedding_layer,
                num_engineered_features=Xf_train.shape[1]
            )

            model.summary(show_trainable=True, line_length=120)

            # Class weights
            class_weight = compute_class_weights(y_train)

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_auc",
                    patience=2,
                    restore_best_weights=True
                )
            ]

            # Training
            model.fit(
                {
                    "q1": q1_train,
                    "q2": q2_train,
                    "engineered_features": Xf_train
                },
                y_train,
                validation_data=(
                    {
                        "q1": q1_test,
                        "q2": q2_test,
                        "engineered_features": Xf_test
                    },
                    y_test
                ),
                batch_size=256,
                epochs=15,
                class_weight=class_weight,
                callbacks=callbacks
            )

            # Evaluation
            metrics = model.evaluate(
                {
                    "q1": q1_test,
                    "q2": q2_test,
                    "engineered_features": Xf_test
                },
                y_test
            )

            logging.info(f"[TRAINER] Test metrics: {metrics}")

            # Save model
            model.save(self.config.model_file_path)
            logging.info(
                f"[TRAINER] Model saved at {self.config.model_file_path}"
            )

            return model

        except Exception as e:
            logging.exception("[TRAINER] Error during model training")
            raise CustomException(e, sys)


# LOCAL TEST
if __name__ == "__main__":
    from src.components.data_ingestion import (
        DataIngestion,
        DataIngestionConfig
    )
    from src.components.data_transformation import DataTransformation

    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    transformer = DataTransformation()

    X_train_df, X_test_df, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path
    )

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(
        X_train_df, X_test_df, y_train.values, y_test.values
    )
