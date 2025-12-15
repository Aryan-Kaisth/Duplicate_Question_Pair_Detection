import os, sys
import keras
import tensorflow as tf
from keras.models import load_model
from src.logger import logging
from src.exception import CustomException

class PredictionPipeline:
    def __init__(self):
        try:
            keras.config.enable_unsafe_deserialization()

            model_path = os.path.join(
                "artifacts", "model_trainer", "siamese.keras"
            )

            self.model = load_model(model_path)
            logging.info("✅ PredictionPipeline initialized successfully.")

        except Exception as e:
            logging.error("❌ Error initializing PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, question1: str, question2: str) -> float:
        q1 = tf.constant([question1], dtype=tf.string)
        q2 = tf.constant([question2], dtype=tf.string)

        score = self.model.predict(
            {"q1": q1, "q2": q2}
        )[0][0]

        return float(score)
