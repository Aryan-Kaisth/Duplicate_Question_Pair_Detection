import os
import sys
from dataclasses import dataclass
from keras.callbacks import EarlyStopping
from src.logger import logging
from src.exception import CustomException
from src.utils.model_utils import (
    build_text_vectorizer,
    build_fasttext_embedding_layer,
    build_siamese_cnn_model,
    compute_class_weights
)

# CONFIG
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join(
        "artifacts", "model_trainer", "custom_model.keras"
    )

# TRAINER
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        logging.info(
            f"[INIT] ModelTrainer ready | Model path: {self.config.model_path}"
        )

    # -----------------------------------------------------
    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("[TRAINER] Training pipeline started")

            # -------- Extract text --------
            q1_train = X_train["question1"].values
            q2_train = X_train["question2"].values
            q1_test  = X_test["question1"].values
            q2_test  = X_test["question2"].values

            logging.info(
                f"[DATA] Train size={len(q1_train)}, Test size={len(q1_test)}"
            )

            # -------- Vectorizer --------
            logging.info("[VECTORIZER] Building text vectorizer")
            vectorizer = build_text_vectorizer(q1_train, q2_train)

            # -------- Embeddings --------
            logging.info("[EMBEDDING] Initializing embedding layer")
            embedding_layer = build_fasttext_embedding_layer(
                vectorizer=vectorizer
            )

            # -------- Model --------
            logging.info("[MODEL] Building Siamese model")
            model = build_siamese_cnn_model(
                vectorizer=vectorizer,
                embedding_layer=embedding_layer
            )

            model.summary(line_length=120)

            # -------- Training setup --------
            class_weight = compute_class_weights(y_train)
            callbacks = [
                EarlyStopping(
                    monitor="val_auc",
                    patience=2,
                    restore_best_weights=True
                )
            ]

            logging.info("[TRAINING] Starting model fit")

            # -------- Train --------
            model.fit(
                {"q1": q1_train, "q2": q2_train},
                y_train,
                validation_data=(
                    {"q1": q1_test, "q2": q2_test},
                    y_test
                ),
                batch_size=256,
                epochs=5,
                callbacks=callbacks,
                verbose=1,
                class_weight=class_weight
            )

            # -------- Evaluate --------
            metrics = model.evaluate(
                {"q1": q1_test, "q2": q2_test},
                y_test,
                verbose=0
            )

            logging.info(f"[EVALUATION] Test metrics: {metrics}")

            # -------- Save --------
            model.save(self.config.model_path)
            logging.info("[SAVE] Model saved successfully")

            logging.info("[TRAINER] Training pipeline completed")

            return model

        except Exception as e:
            logging.exception("[TRAINER] Training pipeline failed")
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