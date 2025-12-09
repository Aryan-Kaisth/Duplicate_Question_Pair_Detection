from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object, read_yaml_file, read_csv_file
import os, sys
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

class CustomPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, text_cols):
        """
        text_cols: list of text column names
        Example: ['question1', 'question2']
        """
        self.text_cols = text_cols
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        # No fitting required for text cleaning
        return self

    def transform(self, X):
        try:
            logging.info("[PIPELINE] Starting CustomPreprocessor")

            X = X.copy()

            for col in self.text_cols:

                # 1. Lowercasing
                X[col] = X[col].str.lower()

                # 2. Remove URLs
                X[col] = X[col].apply(
                    lambda text: re.sub(r'http\S+|www\S+|https\S+', '', text)
                )

                # 3. Expand contractions
                X[col] = X[col].apply(
                    lambda text: contractions.fix(text)
                )

                # 4. Remove accents / diacritics
                X[col] = X[col].apply(
                    lambda text: ''.join(
                        c for c in unicodedata.normalize('NFKD', text)
                        if not unicodedata.combining(c)
                    )
                )

                # 5. Convert emojis to text
                X[col] = X[col].apply(
                    lambda text: emoji.demojize(text)
                )

                # 6. Remove punctuation
                X[col] = X[col].apply(
                    lambda text: text.translate(
                        str.maketrans('', '', string.punctuation)
                    )
                )

                # 7. Tokenization
                X[col] = X[col].apply(
                    lambda text: word_tokenize(text)
                )

                # 8. Lemmatization
                X[col] = X[col].apply(
                    lambda tokens: [self.lemmatizer.lemmatize(word) for word in tokens]
                )

            logging.info("[PIPELINE] CustomPreprocessor completed successfully")
            return X

        except Exception as e:
            logging.error("[ERROR] Error in CustomPreprocessor")
            raise CustomException(e, sys)

class DataTransformationConfig:
    def __init__(self):
        self.custom_preprocessor_path: str = os.path.join('artifacts', 'data_transformation', 'custom_preprocessor.pkl')
        self.schema_path: str = os.path.join('config', 'schema.yaml')

class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()) -> None:
        self._config = config

        os.makedirs(os.path.dirname(self._config.custom_preprocessor_path), exist_ok=True)
        logging.info(f"[INIT] Data transformation artifacts directory ensured.")

        self._schema: dict[str, Any] = read_yaml_file(self._config.schema_path)

        self._target_cols: List[str] = self._schema.get("target_cols", [])
        self._text_cols: List[str] = self._schema.get("text_cols", [])
        self._drop_cols: List[str] = self._schema.get("drop_cols", [])

    def _feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop(columns=self._drop_cols, axis=1)
            df = df.dropna().reset_index(drop=True)
            return df

        except Exception as e:
            raise CustomException(e, sys)
    

    def _split_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X = df[self._text_cols].copy()
            y = df[self._target_cols].copy()
            return X, y

        except Exception as e:
            raise CustomException(e, sys)

    def _save_preprocessor(self, preprocessor: CustomPreprocessor) -> None:
        try:
            save_object(self._config.custom_preprocessor_path, preprocessor)
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            train_df = self._feature_cleaning(train_df)
            test_df = self._feature_cleaning(test_df)

            preprocessor = CustomPreprocessor(text_cols=self._text_cols)

            train_df = preprocessor.transform(train_df)
            test_df = preprocessor.transform(test_df)

            X_train_df, y_train_df = self._split_features_targets(train_df)
            X_test_df, y_test_df = self._split_features_targets(test_df)

            self._save_preprocessor(preprocessor)

            return (
                X_train_df.values,
                X_test_df.values,
                y_train_df.values,
                y_test_df.values,
            )

        except Exception as e:
            raise CustomException(e, sys)



# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig

    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    transformer = DataTransformation()

    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path
    )

    print("✅ X_train:", X_train.shape)
    print("✅ X_test:", X_test.shape)
    print("✅ y_train:", y_train.shape)
    print("✅ y_test:", y_test.shape)

