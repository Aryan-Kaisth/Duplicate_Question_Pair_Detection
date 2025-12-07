# data_ingestion.py

import os
import sys
from dataclasses import dataclass
from typing import Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_csv_file, read_yaml_file
from database.connection import get_connection
from database.queries import get_all_data


@dataclass
class DataIngestionConfig:
    """
    Holds all configurable paths needed during the data ingestion step.
    These paths help me keep the ingestion outputs organized.
    """
    raw_data_dir: str = os.path.join("artifacts", "data_ingestion")
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    table_name_path: str = os.path.join("config", "database.yaml")


class DataIngestion:
    """
    Handles the entire ingestion workflow.
    Only one public method `initiate_data_ingestion()` should be used.
    Everything else is private and not meant to be accessed outside.
    """

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()) -> None:
        """
        Initialize ingestion with config and load YAML information.
        I also ensure the raw-data folder exists before we start writing.
        """
        self._config = config
        self._table_name_config: str = read_yaml_file(config.table_name_path)

        os.makedirs(self._config.raw_data_dir, exist_ok=True)
        logging.info(f"[INIT] Ensured data ingestion directory: {self._config.raw_data_dir}")

    # PRIVATE METHODS (internal helpers)

    def _get_table_name(self) -> str:
        """
        Internal helper to fetch the correct table name from YAML.
        """
        table_name = self._table_name_config.get("table")

        if not table_name:
            raise ValueError("table not found in database.yaml")

        return table_name

    def _fetch_data_from_db(self) -> pd.DataFrame:
        """
        Internal method for connecting to DB and loading the entire table into a DataFrame.
        """
        try:
            table_name = self._get_table_name()
            logging.info(f"[DB] Fetching data from table: {table_name}")

            with get_connection() as conn:
                stmt = get_all_data(table_name)
                result = conn.execute(stmt)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            logging.info(f"[DB] Data fetched → rows: {df.shape[0]}, columns: {df.shape[1]}")
            return df

        except Exception as e:
            logging.error("[ERROR] Database fetch failed.")
            raise CustomException(e, sys)

    def _save_raw_data(self, df: pd.DataFrame) -> None:
        """
        Internal method to save the raw, untouched dataset.
        """
        save_csv_file(df, self._config.raw_data_path)
        logging.info(f"[SAVE] Raw data stored at {self._config.raw_data_path}")

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Internal method to split the dataset into train and test sets.
        """
        logging.info("[SPLIT] Performing train-test split (80-20)...")
        train_set, test_set = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )
        logging.info(f"[SPLIT] Train: {train_set.shape}, Test: {test_set.shape}")
        return train_set, test_set

    def _save_splits(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Internal method to save train and test datasets.
        """
        save_csv_file(train, self._config.train_data_path)
        logging.info(f"[SAVE] Train data saved to {self._config.train_data_path}")

        save_csv_file(test, self._config.test_data_path)
        logging.info(f"[SAVE] Test data saved to {self._config.test_data_path}")

    # PUBLIC METHOD (single entry point)

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Public method that orchestrates the ingestion pipeline.
        This is the only method the outside world should use.
        """
        logging.info("==== DATA INGESTION STARTED ====")

        try:
            df = self._fetch_data_from_db()
            self._save_raw_data(df)

            train_set, test_set = self._split_data(df)
            self._save_splits(train_set, test_set)

            logging.info("==== DATA INGESTION COMPLETED SUCCESSFULLY ====")

            return self._config.train_data_path, self._config.test_data_path

        except Exception as e:
            logging.error("[FATAL] Ingestion pipeline failed.")
            raise CustomException(e, sys)


# Allow running this file directly for debugging
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"✔ Train data saved to: {train_path}")
    print(f"✔ Test data saved to: {test_path}")