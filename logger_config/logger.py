# logger_config/logger.py

import logging
import os
from datetime import datetime

def get_logger(name=__name__):
    os.makedirs("logs", exist_ok=True)
    log_filename = datetime.now().strftime("logs/pipeline_%Y%m%d_%H%M%S.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler()

        # ✨ Updated Formatter to include filename
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(filename)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
