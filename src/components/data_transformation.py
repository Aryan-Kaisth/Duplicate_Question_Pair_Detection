import os
import sys
from typing import List, Tuple, Any

import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import (
    save_object,
    read_yaml_file,
    read_csv_file
)
from src.utils.custom_preprocessor import CustomPreprocessor


# CONFIG
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts", "data_transformation", "custom_preprocessor.pkl"
        )
        self.schema_path = os.path.join("config", "schema.yaml")


# DATA TRANSFORMATION
class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

        os.makedirs(
            os.path.dirname(self.config.preprocessor_path),
            exist_ok=True
        )

        logging.info(
            "[INIT] DataTransformation initialized | Loading schema configuration"
        )

        schema = read_yaml_file(self.config.schema_path)

        self.target_cols: List[str] = schema.get("target_cols", [])
        self.text_cols: List[str] = schema.get("text_cols", [])
        self.drop_cols: List[str] = schema.get("drop_cols", [])

        logging.info(
            f"[INIT] Schema loaded | "
            f"Targets={self.target_cols}, "
            f"TextCols={self.text_cols}, "
            f"DropCols={self.drop_cols}"
        )

    # -----------------------------------------------------
    def _clean_dataframe(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Basic cleaning: drop unwanted columns and NA rows."""
        logging.info(
            f"[CLEAN:{stage}] Starting data cleaning | Shape={df.shape}"
        )

        df = df.drop(columns=self.drop_cols, errors="ignore")
        df = df.dropna().reset_index(drop=True)

        logging.info(
            f"[CLEAN:{stage}] Cleaning completed | Shape={df.shape}"
        )
        return df

    # -----------------------------------------------------
    def _split_X_y(
        self, df: pd.DataFrame, stage: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        logging.info(
            f"[SPLIT:{stage}] Splitting features and target columns"
        )

        X = df.drop(columns=self.target_cols)
        y = df[self.target_cols]

        logging.info(
            f"[SPLIT:{stage}] Split completed | "
            f"X={X.shape}, y={y.shape}"
        )
        return X, y

    # -----------------------------------------------------
    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        try:
            logging.info(
                "[DataTransformation] Pipeline started"
            )

            # -------- Load data --------
            logging.info(
                f"[LOAD] Reading train data from: {train_path}"
            )
            train_df = read_csv_file(train_path)

            logging.info(
                f"[LOAD] Reading test data from: {test_path}"
            )
            test_df = read_csv_file(test_path)

            logging.info(
                f"[LOAD] Data loaded | "
                f"TrainShape={train_df.shape}, TestShape={test_df.shape}"
            )

            # -------- Clean data --------
            train_df = self._clean_dataframe(train_df, stage="TRAIN")
            test_df = self._clean_dataframe(test_df, stage="TEST")

            # -------- Text preprocessing --------
            logging.info(
                "[PREPROCESS] Initializing CustomPreprocessor for text columns"
            )

            preprocessor = CustomPreprocessor(text_cols=self.text_cols)

            logging.info(
                "[PREPROCESS] Fitting preprocessor on training data"
            )
            train_df = preprocessor.fit_transform(train_df)

            logging.info(
                "[PREPROCESS] Applying preprocessor on test data"
            )
            test_df = preprocessor.transform(test_df)

            # -------- Split features / targets --------
            X_train, y_train = self._split_X_y(train_df, stage="TRAIN")
            X_test, y_test = self._split_X_y(test_df, stage="TEST")

            # -------- Save preprocessor --------
            save_object(self.config.preprocessor_path, preprocessor)
            logging.info(
                f"[SAVE] Preprocessor artifact saved | "
                f"Path={self.config.preprocessor_path}"
            )

            logging.info(
                "[DataTransformation] Pipeline completed successfully"
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.exception(
                "[DataTransformation] Pipeline failed due to an unexpected error"
            )
            raise CustomException(e, sys)


# LOCAL TEST
if __name__ == "__main__":
    from src.components.data_ingestion import (
        DataIngestion,
        DataIngestionConfig
    )

    ingestion = DataIngestion(DataIngestionConfig())
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path, test_path
    )

    print("✅ X_train:", X_train.shape)
    print("✅ X_test :", X_test.shape)
    print("✅ y_train:", y_train.shape)
    print("✅ y_test :", y_test.shape)
