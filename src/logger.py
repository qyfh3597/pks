"""
Logging configuration for the Personal Knowledge Summary System.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "PersonalSummary",
    log_level: str = "INFO",
    log_file: str = None,
    log_dir: str = "./output"
) -> logging.Logger:
    """
    Setup logger for the system.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set log file path
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"system_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized: {log_file}")
    
    return logger


# Global logger instance
_logger_instance = None


def get_logger(name: str = "PersonalSummary") -> logging.Logger:
    """
    Get global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = setup_logger(name)
    return _logger_instance
