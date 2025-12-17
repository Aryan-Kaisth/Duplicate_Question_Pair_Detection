import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def run(self):
        try:
            logging.info("===== TRAINING PIPELINE STARTED =====")

            # Data Ingestion
            logging.info("[INGESTION] Starting data ingestion")

            ingestion = DataIngestion(DataIngestionConfig())
            train_path, test_path = ingestion.initiate_data_ingestion()

            logging.info(
                f"[INGESTION] Completed | "
                f"train_path={train_path}, test_path={test_path}"
            )

            # Data Transformation
            logging.info("[TRANSFORMATION] Starting data transformation")

            transformer = DataTransformation()
            X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path
            )

            logging.info(
                f"[TRANSFORMATION] Completed | "
                f"X_train={X_train.shape}, X_test={X_test.shape}, "
                f"y_train={y_train.shape}, y_test={y_test.shape}"
            )

            # Model Training
            logging.info("[TRAINING] Starting model training")

            trainer = ModelTrainer()
            model = trainer.initiate_model_trainer(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

            logging.info("[TRAINING] Model training completed")

            logging.info("===== TRAINING PIPELINE COMPLETED SUCCESSFULLY =====")
            return model

        except Exception as e:
            logging.exception(
                "[PIPELINE] Training pipeline failed due to an unexpected error"
            )
            raise CustomException(e, sys)
