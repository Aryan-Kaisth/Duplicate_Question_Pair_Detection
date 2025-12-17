import os
import sys
import tensorflow as tf
from keras.models import load_model
from src.logger import logging
from src.exception import CustomException
from src.utils.custom_preprocessor import CustomPreprocessor


class PredictionPipeline:
    def __init__(self):
        try:
            model_path = os.path.join(
                "artifacts", "model_trainer", "custom_model.keras"
            )

            logging.info(f"[INIT] Loading model from {model_path}")
            self.model = load_model(model_path)

            self.preprocessor = CustomPreprocessor()

            logging.info("[INIT] PredictionPipeline ready")

        except Exception as e:
            logging.exception("[INIT] Model loading failed")
            raise CustomException(e, sys)

    def predict(self, question1: str, question2: str) -> float:
        try:
            logging.info("[PREDICT] Raw input received")
            logging.info(f"[PREDICT] Question 1 (raw): {question1}")
            logging.info(f"[PREDICT] Question 2 (raw): {question2}")

            q1 = self.preprocessor.transform(question1)
            q2 = self.preprocessor.transform(question2)

            logging.info("[PREDICT] Preprocessing completed")
            logging.info(f"[PREDICT] Question 1 (cleaned): {q1}")
            logging.info(f"[PREDICT] Question 2 (cleaned): {q2}")

            score = self.model.predict(
                (
                    tf.constant([q1], dtype=tf.string),
                    tf.constant([q2], dtype=tf.string),
                ),
                verbose=0
            )[0][0]

            logging.info(f"[PREDICT] Prediction score: {float(score)}")

            return float(score)

        except Exception as e:
            logging.exception("[PREDICT] Inference failed")
            raise CustomException(e, sys)
