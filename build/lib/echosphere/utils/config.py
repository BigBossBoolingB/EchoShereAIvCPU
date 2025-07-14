"""
Configuration management for EchoSphere AI-vCPU.

This module provides configuration classes and utilities for managing
system parameters, database connections, and runtime settings.
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Config:
    """
    EchoSphere system configuration.

    Manages all configuration parameters for the system including
    database connections, logging, and performance settings.
    """

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    vsa_dimensions: int = 10000
    vsa_device: str = "cpu"
    vsa_type: str = "BSC"

    actor_timeout: float = 5.0
    workspace_max_messages: int = 1000
    workspace_max_history: int = 5000

    max_concurrent_tasks: int = 10
    query_timeout: float = 30.0
    similarity_threshold: float = 0.7

    data_dir: Path = field(default_factory=lambda: Path.home() / ".echosphere")
    log_dir: Path = field(default_factory=lambda: Path.home() / ".echosphere" / "logs")

    def __post_init__(self):
        """Post-initialization setup."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_username = os.getenv("NEO4J_USERNAME", self.neo4j_username)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_database = os.getenv("NEO4J_DATABASE", self.neo4j_database)

        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

        vsa_dims = os.getenv("VSA_DIMENSIONS")
        if vsa_dims:
            self.vsa_dimensions = int(vsa_dims)

        self.vsa_device = os.getenv("VSA_DEVICE", self.vsa_device)
        self.vsa_type = os.getenv("VSA_TYPE", self.vsa_type)

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Config object loaded from file
        """
        config = cls()

        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def to_file(self, config_path: Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path where to save configuration
        """
        config_data = {}

        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_data[key] = str(value)
            else:
                config_data[key] = value

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or environment.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Loaded configuration object
    """
    if config_path is None:
        possible_paths = [
            Path.cwd() / "echosphere.yaml",
            Path.cwd() / "config.yaml",
            Path.home() / ".echosphere" / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path and config_path.exists():
        return Config.from_file(config_path)
    else:
        return Config()
