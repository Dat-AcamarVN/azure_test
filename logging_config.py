"""
Logging Configuration for Patent Search System
Control logging levels for different components
"""

import logging


def configure_logging(log_level: str = "INFO", disable_azure_logs: bool = True):
    """
    Configure logging for the entire system
    
    Args:
        log_level: Main logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        disable_azure_logs: Whether to disable verbose Azure SDK logging
    """
    
    # Set main logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if disable_azure_logs:
        # Disable verbose HTTP logging from Azure SDK
        azure_loggers = [
            "azure.core.pipeline.policies.http_logging_policy",
            "azure.cosmos",
            "azure.search", 
            "azure.identity",
            "azure.core.pipeline",
            "azure.core.pipeline.transport",
            "azure.core.rest",
            "azure.core.credentials",
            "azure.core.auth",
            "azure.core.exceptions"
        ]
        
        for logger_name in azure_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # Also disable some other verbose loggers
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        logging.getLogger("urllib3.util.retry").setLevel(logging.WARNING)
        logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)
    
    # Set our custom loggers to desired level
    logging.getLogger("utilities.chunking_utils").setLevel(numeric_level)
    logging.getLogger("dao.patent_dao").setLevel(numeric_level)
    logging.getLogger("models.patent_model").setLevel(numeric_level)
    
    print(f"✅ Logging configured: Level={log_level}, Azure logs disabled={disable_azure_logs}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


# Preset configurations
def configure_test_logging():
    """Configure logging specifically for testing - minimal output"""
    configure_logging("INFO", disable_azure_logs=True)


def configure_debug_logging():
    """Configure logging for debugging - more detailed output"""
    configure_logging("DEBUG", disable_azure_logs=False)


def configure_production_logging():
    """Configure logging for production - only important messages"""
    configure_logging("WARNING", disable_azure_logs=True)
