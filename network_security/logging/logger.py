import logging
import os
from datetime import datetime

# Create file name structure and path
LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log"
LOGS_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create a directory to store logs if it does not exist
os.makedirs(LOGS_PATH, exist_ok=True)

# Create the path for the log files
LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
