import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded CSV data.

    Raises
    ------
    CustomException
        If the file does not exist or cannot be read.
    """
    try:
        logging.info(f"[READ CSV] Attempting to read CSV file at: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at: {file_path}")

        df = pd.read_csv(file_path)

        logging.info(f"[READ CSV] Successfully read CSV file. Shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"[READ CSV] Failed to read CSV file: {file_path}")
        raise CustomException(e, sys)


def save_csv_file(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to save.
    file_path : str
        Output CSV path.

    Raises
    ------
    CustomException
        For any write or path errors.
    """
    try:
        logging.info(f"[SAVE CSV] Saving DataFrame to: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)

        logging.info(f"[SAVE CSV] DataFrame successfully saved. Shape: {data.shape}")

    except Exception as e:
        logging.error(f"[SAVE CSV] Failed to save DataFrame to: {file_path}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a Python object from a serialized joblib file.

    Parameters
    ----------
    file_path : str
        Path to the joblib file.

    Returns
    -------
    object
        Loaded Python object.

    Raises
    ------
    CustomException
        If the file cannot be loaded.
    """
    try:
        logging.info(f"[LOAD OBJ] Loading object from: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)

        logging.info("[LOAD OBJ] Object successfully loaded.")
        return obj

    except Exception as e:
        logging.error(f"[LOAD OBJ] Failed to load object: {file_path}")
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save a NumPy array to a .npy file.

    Parameters
    ----------
    file_path : str
        Destination file path.
    array : np.ndarray
        NumPy array to save.

    Raises
    ------
    CustomException
        In case of write failure.
    """
    try:
        logging.info(f"[SAVE NUMPY] Saving NumPy array to: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logging.info(f"[SAVE NUMPY] Array saved successfully. Shape: {array.shape}")

    except Exception as e:
        logging.error(f"[SAVE NUMPY] Failed to save NumPy array to: {file_path}")
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load NumPy array data from a .npy file.

    Parameters
    ----------
    file_path : str
        File path of the .npy file.

    Returns
    -------
    np.ndarray
        Loaded NumPy array.

    Raises
    ------
    CustomException
        If the array cannot be read.
    """
    try:
        logging.info(f"[LOAD NUMPY] Loading array from: {file_path}")

        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)

        logging.info(f"[LOAD NUMPY] Array loaded successfully. Shape: {array.shape}")
        return array

    except Exception as e:
        logging.error(f"[LOAD NUMPY] Failed to load array from: {file_path}")
        raise CustomException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object using joblib serialization.

    Parameters
    ----------
    file_path : str
        Destination file path.
    obj : object
        The Python object to serialize.

    Raises
    ------
    CustomException
        If saving fails.
    """
    try:
        logging.info(f"[SAVE OBJ] Saving object to: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

        logging.info("[SAVE OBJ] Object successfully saved.")

    except Exception as e:
        logging.error(f"[SAVE OBJ] Failed to save object to: {file_path}")
        raise CustomException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML configuration file.

    Parameters
    ----------
    file_path : str
        YAML file path.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    CustomException
        If the file cannot be read or parsed.
    """
    try:
        logging.info(f"[READ YAML] Reading YAML file: {file_path}")

        with open(file_path, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)

        logging.info("[READ YAML] YAML file read successfully.")
        return content

    except Exception as e:
        logging.error(f"[READ YAML] Failed to read YAML file: {file_path}")
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write a Python object to a YAML file.

    Parameters
    ----------
    file_path : str
        Destination YAML file path.
    content : object
        The content to write.
    replace : bool, optional
        Whether to replace the existing file (default False).

    Raises
    ------
    CustomException
        If the file cannot be written.
    """
    try:
        logging.info(f"[WRITE YAML] Writing YAML file: {file_path} | replace={replace}")

        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

        logging.info("[WRITE YAML] YAML file written successfully.")

    except Exception as e:
        logging.error(f"[WRITE YAML] Failed to write YAML file: {file_path}")
        raise CustomException(e, sys)