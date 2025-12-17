import re
import string
import unicodedata
import emoji
import sys
import contractions
from src.logger import logging
from src.exception import CustomException
from sklearn.base import TransformerMixin, BaseEstimator
from typing import List, Any, Tuple
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, text_cols=None):
        self.text_cols = text_cols
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        logging.info("[CustomPreprocessor] Fit called (stateless)")
        return self

    def _clean_text(self, text: str) -> str:
        text = str(text).lower()

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = contractions.fix(text)

        text = "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )

        text = emoji.demojize(text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens]

        return " ".join(tokens)

    def transform(self, X):
        logging.info("[CustomPreprocessor] Transform called")

        try:
            if isinstance(X, str):
                return self._clean_text(X)


            if isinstance(X, (list, tuple)):
                return [self._clean_text(x) for x in X]

            if isinstance(X, pd.DataFrame):
                if not self.text_cols:
                    raise ValueError(
                        "text_cols must be provided when transforming a DataFrame"
                    )

                X = X.copy()

                for col in self.text_cols:
                    logging.info(f"[CustomPreprocessor] Processing column: {col}")
                    X[col] = X[col].apply(self._clean_text)

                return X

            raise TypeError(
                f"Unsupported input type: {type(X)}"
            )

        except Exception as e:
            logging.exception("[CustomPreprocessor] Error during transform")
            raise CustomException(e, sys)
