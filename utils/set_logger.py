import logging
from datetime import datetime
from typing import Union

def MyLogger(setLevel : str = 'info', to_file:Union[str, None] = None) -> logging.Logger:
    """
    Create a logger instance that outputs logs to the console and optionally to a daily log file.
    
    Args:
        setLevel (str): Logging level ('info', 'debug', 'error', 'warn', 'warning').
        to_file (Union[str, None]): Custom log file name. If None, defaults to a daily file based on the current date.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    
    # Create logger
    logger = logging.getLogger()

    # Ensure no duplicate handlers
    if logger.hasHandlers():
        return logger

    # Set logging format 
    formatter = logging.Formatter('%(asctime)s - %(module)s> %(funcName)s> %(lineno)s - %(levelname)s - %(message)s')

    # Set logging level
    match setLevel.upper():
        case 'INFO': logger.setLevel(logging.INFO)
        case 'DEBUG': logger.setLevel(logging.DEBUG)
        case 'ERROR': logger.setLevel(logging.ERROR)
        case 'WARN' | 'WARNING': logger.setLevel(logging.WARNING)
        case _: logger.setLevel(logging.INFO)  # Default level

    # Add stream handler (console output)        
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Add file handler (file output)
    if to_file is None:
        # Generate a daily log file name
        current_date = datetime.now().date().isoformat()
        to_file = f'log_{current_date}.log'
    
    file_handler = logging.FileHandler(to_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
