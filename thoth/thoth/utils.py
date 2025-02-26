import os
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from dotenv import load_dotenv
import sys

# Import the custom logger from the project
try:
    from terrorblade.utils.logger import Logger, NICE
except ImportError:
    # Define NICE level if we can't import it
    NICE = 25
    if not hasattr(logging, 'NICE'):
        logging.addLevelName(NICE, 'NICE')
        def nice(self, message, *args, **kwargs):
            if self.isEnabledFor(NICE):
                self._log(NICE, message, args, **kwargs)
        logging.Logger.nice = nice

class Config:
    """Configuration handler for Thoth that loads from environment variables or .env file."""
    
    def __init__(self, env_path: Optional[str | Path] = None):
        """
        Initialize configuration from environment variables or .env file.
        
        Args:
            env_path: Path to .env file, defaults to .env in current directory
        """
        # Load environment variables from .env file if it exists
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        # Database settings
        self.db_path = os.getenv("THOTH_DB_PATH", "telegram_data.db")
        self.phone = os.getenv("THOTH_PHONE", None)
        
        # Qdrant settings
        self.qdrant_mode = os.getenv("THOTH_QDRANT_MODE", "local").lower()
        self.qdrant_path = os.getenv("THOTH_QDRANT_PATH", "./qdrant_db")
        self.qdrant_url = os.getenv("THOTH_QDRANT_URL", None)
        self.qdrant_api_key = os.getenv("THOTH_QDRANT_API_KEY", None)
        
        # Embedding model settings
        self.embedding_model = os.getenv("THOTH_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vector_size = int(os.getenv("THOTH_VECTOR_SIZE", "384"))
        
        # Logging settings
        self.log_level = os.getenv("THOTH_LOG_LEVEL", "INFO")
        self.log_file = os.getenv("THOTH_LOG_FILE", None)
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging with appropriate handlers."""
        try:
            # Try to use the project's logger first
            self.logger = Logger(
                name="thoth",
                level=self.log_level,
                file=self.log_file
            )
        except Exception as e:
            # Fall back to standard logging
            self.logger = logging.getLogger("thoth")
            self.logger.setLevel(getattr(logging, self.log_level))
            
            # Clear existing handlers
            if self.logger.handlers:
                self.logger.handlers.clear()
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.log_level))
            
            # Create formatter
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
            
            # Add file handler if specified
            if self.log_file:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(getattr(logging, self.log_level))
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """
        Get Qdrant client configuration based on mode.
        
        Returns:
            Dict with Qdrant client configuration parameters
        """
        if self.qdrant_mode == "remote" and self.qdrant_url:
            config = {"url": self.qdrant_url}
            if self.qdrant_api_key:
                config["api_key"] = self.qdrant_api_key
            self.logger.info(f"Using remote Qdrant at: {self.qdrant_url}")
            return config
        else:
            self.logger.info(f"Using local Qdrant at path: {self.qdrant_path}")
            return {"path": self.qdrant_path}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "db_path": self.db_path,
            "phone": self.phone,
            "qdrant_mode": self.qdrant_mode,
            "qdrant_path": self.qdrant_path,
            "qdrant_url": self.qdrant_url,
            "embedding_model": self.embedding_model,
            "vector_size": self.vector_size,
            "log_level": self.log_level,
            "log_file": self.log_file
        } 