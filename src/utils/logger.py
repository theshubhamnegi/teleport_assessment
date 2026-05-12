import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('LEVEL=%(levelname)s | FUNCTION=%(funcName)s | MODULE=%(name)s | FILE=%(filename)s:%(lineno)d | MESSAGE=%(message)s')
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
