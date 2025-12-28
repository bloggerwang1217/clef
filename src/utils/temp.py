"""
Temporary file management utilities.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TempFileManager:
    """
    Manages temporary files and directories with automatic cleanup.

    Usage:
        with TempFileManager("./temp") as tmp:
            wav_path = tmp.get_path("audio.wav")
            # ... use wav_path ...
        # Automatically cleaned up
    """

    def __init__(self, base_dir: str = "./temp"):
        """
        Initialize temporary file manager.

        Args:
            base_dir: Base directory for temporary files
        """
        self.base_dir = Path(base_dir)
        self._created = False

    def __enter__(self) -> "TempFileManager":
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    def create(self) -> Path:
        """
        Create the temporary directory.

        Returns:
            Path to the temporary directory
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._created = True
        logger.debug(f"Created temporary directory: {self.base_dir}")
        return self.base_dir

    def cleanup(self) -> None:
        """Remove all temporary files and the directory."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
            logger.info(f"Cleaned up temporary directory: {self.base_dir}")
            self._created = False

    def get_path(self, filename: str) -> Path:
        """
        Get a path within the temporary directory.

        Args:
            filename: Name of the file

        Returns:
            Full path to the file
        """
        if not self._created:
            self.create()
        return self.base_dir / filename

    def get_subdir(self, dirname: str) -> Path:
        """
        Get or create a subdirectory within the temporary directory.

        Args:
            dirname: Name of the subdirectory

        Returns:
            Path to the subdirectory
        """
        if not self._created:
            self.create()
        subdir = self.base_dir / dirname
        subdir.mkdir(exist_ok=True)
        return subdir

    @property
    def path(self) -> Path:
        """Get the base temporary directory path."""
        return self.base_dir
