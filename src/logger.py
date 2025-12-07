import os
import logging
from datetime import datetime

# Original log file name
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Folder name WITHOUT .log
FOLDER_NAME = LOG_FILE.replace(".log", "")

# logs/<timestamp>/
log_path = os.path.join(os.getcwd(), "logs", FOLDER_NAME)
os.makedirs(log_path, exist_ok=True)

# logs/<timestamp>/<timestamp>.log
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
