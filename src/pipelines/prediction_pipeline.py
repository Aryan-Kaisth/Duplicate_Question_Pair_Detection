import os, sys
import tensorflow as tf
from src.logger import logging
from src.exception import CustomException
from keras.models import load_model


class PredictionPipeline:
    def __init__(self):
        """
        Initializes the prediction pipeline by loading the trained model.
        """
        try:
            model_path = os.path.join(
                "artifacts", "model_trainer", "siamese.keras"
            )

            self.model = load_model(model_path)
            logging.info("✅ PredictionPipeline initialized successfully.")

        except Exception as e:
            logging.error("❌ Error initializing PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, question1: str, question2: str):
        """
        Generates similarity prediction for two input questions.

        Args:
            question1 (str): First question
            question2 (str): Second question

        Returns:
            float: Similarity score (0–1)
        """
        try:
            logging.info(f"Q1 -> {question1}")
            logging.info(f"Q2 -> {question2}")

            # Convert to tf.string tensors
            q1 = tf.constant([question1], dtype=tf.string)
            q2 = tf.constant([question2], dtype=tf.string)

            # Predict similarity
            preds = self.model.predict(
                {"q1": q1, "q2": q2}
            )

            score = float(preds[0][0])
            logging.info(f"✅ Similarity score: {score}")

            return score

        except Exception as e:
            logging.error("❌ Error during prediction.")
            raise CustomException(e, sys)
