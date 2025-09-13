import os
import sys
import logging  
import datetime

logging_str = "[%(asctime)s|(%(levelname)s)| File: %(module)s | Message: %(message)s]"

# get current date
today_date = datetime.datetime.now().strftime("%d-%m-%Y")
log_dir = "logs"
log_file = os.path.join(log_dir, f"{today_date}.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("medical_chatbot_logger")