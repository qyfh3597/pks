"""
Configuration management for the Personal Knowledge Summary System.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration class for managing system settings."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration from JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., "data.raw_data_dir" or "embedding.model_name")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., "data.raw_data_dir")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        path = output_path or self.config_path
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def ensure_directories(self) -> None:
        """Create all necessary directories defined in configuration."""
        dirs_to_create = [
            self.get("data.raw_data_dir"),
            self.get("data.processed_data_dir"),
            self.get("data.labels_dir"),
            self.get("output.root_dir"),
            self.get("output.summaries_dir"),
            self.get("output.reports_dir"),
            self.get("output.models_dir"),
            self.get("output.experiments_dir"),
            self.get("output.paper_dir"),
            self.get("chroma.persist_dir"),
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2, ensure_ascii=False)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.json") -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
