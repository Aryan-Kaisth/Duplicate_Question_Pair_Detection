from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import (
    save_object,
    read_yaml_file,
    read_csv_file,
    fetch_fuzzy_features,
    fetch_token_features,
    common_words,
    total_words
)

import os
import sys
import time
from typing import List, Any, Tuple

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

import re
import string
import unicodedata
import emoji
import contractions

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# CUSTOM TEXT PREPROCESSOR
class CustomPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, text_cols: List[str]):
        self.text_cols = text_cols
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        logging.info("[CustomPreprocessor] Fit called (no state learned)")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        logging.info(
            f"[CustomPreprocessor] Transform started | Rows={X.shape[0]} Cols={X.shape[1]}"
        )

        try:
            X = X.copy()

            for col in self.text_cols:
                logging.info(f"[CustomPreprocessor] Processing column: {col}")

                X[col] = X[col].astype(str).str.lower()

                X[col] = X[col].apply(
                    lambda t: re.sub(r"http\S+|www\S+|https\S+", "", t)
                )

                X[col] = X[col].apply(contractions.fix)

                X[col] = X[col].apply(
                    lambda t: "".join(
                        c for c in unicodedata.normalize("NFKD", t)
                        if not unicodedata.combining(c)
                    )
                )

                X[col] = X[col].apply(emoji.demojize)

                X[col] = X[col].apply(
                    lambda t: t.translate(
                        str.maketrans("", "", string.punctuation)
                    )
                )

                X[col] = X[col].apply(word_tokenize)

                X[col] = X[col].apply(
                    lambda tokens: [
                        self.lemmatizer.lemmatize(tok) for tok in tokens
                    ]
                )

                X[col] = X[col].apply(lambda tokens: " ".join(tokens))

            logging.info(
                f"[CustomPreprocessor] Completed in {time.time() - start_time:.2f}s"
            )
            return X

        except Exception as e:
            logging.exception("[CustomPreprocessor] Error during transform")
            raise CustomException(e, sys)


# CONFIG
class DataTransformationConfig:
    def __init__(self):
        self.custom_preprocessor_path = os.path.join(
            "artifacts", "data_transformation", "custom_preprocessor.pkl"
        )
        self.schema_path = os.path.join("config", "schema.yaml")


# DATA TRANSFORMATION
class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self._config = config

        os.makedirs(
            os.path.dirname(self._config.custom_preprocessor_path),
            exist_ok=True
        )

        logging.info("[INIT] DataTransformation initialized")

        self._schema: dict[str, Any] = read_yaml_file(self._config.schema_path)

        self._target_cols: List[str] = self._schema.get("target_cols", [])
        self._text_cols: List[str] = self._schema.get("text_cols", [])
        self._drop_cols: List[str] = self._schema.get("drop_cols", [])

    # -------------------------------------------------
    def _feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(
                f"[FeatureCleaning] Initial shape: {df.shape}"
            )

            df = df.drop(columns=self._drop_cols, errors="ignore")
            df = df.dropna().reset_index(drop=True)

            logging.info(
                f"[FeatureCleaning] Final shape: {df.shape}"
            )
            return df

        except Exception as e:
            logging.exception("[FeatureCleaning] Failed")
            raise CustomException(e, sys)

    # -------------------------------------------------
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logging.info(
            f"[FeatureEngineering] Started | Rows={df.shape[0]}"
        )

        try:
            df["q1_len"] = df["question1"].str.len()
            df["q2_len"] = df["question2"].str.len()

            df["q1_num_words"] = df["question1"].apply(lambda x: len(x.split()))
            df["q2_num_words"] = df["question2"].apply(lambda x: len(x.split()))

            df["word_common"] = df.apply(common_words, axis=1)
            df["word_total"] = df.apply(total_words, axis=1)
            df["word_share"] = (
                df["word_common"] / df["word_total"].replace(0, 1)
            )

            token_features = df.apply(fetch_token_features, axis=1)

            df["cwc_min"] = token_features.map(lambda x: x[0])
            df["cwc_max"] = token_features.map(lambda x: x[1])
            df["csc_min"] = token_features.map(lambda x: x[2])
            df["csc_max"] = token_features.map(lambda x: x[3])
            df["ctc_min"] = token_features.map(lambda x: x[4])
            df["ctc_max"] = token_features.map(lambda x: x[5])
            df["last_word_eq"] = token_features.map(lambda x: x[6])
            df["first_word_eq"] = token_features.map(lambda x: x[7])

            fuzzy_features = df.apply(fetch_fuzzy_features, axis=1)

            df["fuzz_ratio"] = fuzzy_features.map(lambda x: x[0])
            df["fuzz_partial_ratio"] = fuzzy_features.map(lambda x: x[1])
            df["token_sort_ratio"] = fuzzy_features.map(lambda x: x[2])
            df["token_set_ratio"] = fuzzy_features.map(lambda x: x[3])

            logging.info(
                f"[FeatureEngineering] Completed | Shape={df.shape} | Time={time.time() - start:.2f}s"
            )
            return df

        except Exception as e:
            logging.exception("[FeatureEngineering] Failed")
            raise CustomException(e, sys)

    # -------------------------------------------------
    def _split_features_targets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        try:
            X = df.drop(columns=self._target_cols)
            y = df[self._target_cols]
            return X, y

        except Exception as e:
            logging.exception("[Split] Failed")
            raise CustomException(e, sys)

    # -------------------------------------------------
    def _save_preprocessor(self, preprocessor: CustomPreprocessor) -> None:
        try:
            save_object(self._config.custom_preprocessor_path, preprocessor)
            logging.info(
                f"[SAVE] Preprocessor saved at {self._config.custom_preprocessor_path}"
            )
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ):
        try:
            logging.info("[DataTransformation] Started")

            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            train_df = self._feature_cleaning(train_df)
            test_df = self._feature_cleaning(test_df)

            train_df = self._feature_engineering(train_df)
            test_df = self._feature_engineering(test_df)

            preprocessor = CustomPreprocessor(text_cols=self._text_cols)

            train_df = preprocessor.fit_transform(train_df)
            test_df = preprocessor.transform(test_df)

            X_train_df, y_train_df = self._split_features_targets(train_df)
            X_test_df, y_test_df = self._split_features_targets(test_df)

            self._save_preprocessor(preprocessor)

            logging.info(
                f"[DataTransformation] X_train={X_train_df.shape}, y_train={y_train_df.shape}"
            )
            logging.info(
                f"[DataTransformation] X_test={X_test_df.shape}, y_test={y_test_df.shape}"
            )

            return (
                X_train_df,
                X_test_df,
                y_train_df,
                y_test_df
            )

        except Exception as e:
            logging.exception("[DataTransformation] Failed")
            raise CustomException(e, sys)


# TESTING
if __name__ == "__main__":
    from src.components.data_ingestion import (
        DataIngestion,
        DataIngestionConfig
    )

    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    transformer = DataTransformation()

    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path
    )

    print("✅ X_train:", X_train.shape)
    print("✅ X_test :", X_test.shape)
    print("✅ y_train:", y_train.shape)
    print("✅ y_test :", y_test.shape)
    print(X_train)