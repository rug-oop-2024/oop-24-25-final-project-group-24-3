from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Handles the case when a path is not found.
    """
    def __init__(self, path: str) -> None:
        """
        Initialization of the class.
        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for loading, saving, deleting, or listing paths.
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """Save data to a given path."""
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load data from a given path."""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete data at a given path."""
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """List all paths under a given path."""
        pass


class LocalStorage(Storage):
    """
    Storage implementation for handling files locally.
    """
    def __init__(self, base_path: str = "./assets") -> None:
        """Initialize the local storage with a base directory."""
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """Save data to the specified file path."""
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """Load data from the specified file path."""
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """Delete data at the specified file path."""
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """List all file paths under the specified directory."""
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """Ensure the specified path exists, raising an error if not."""
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Construct a full path from the base path and a relative path."""
        return os.path.join(self._base_path, path)
