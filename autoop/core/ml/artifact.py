from pydantic import BaseModel, Field
from typing import List, Optional
import base64


class Artifact(BaseModel):
    """Represents an artifact with essential metadata and binary data."""

    type: str = Field(...)
    name: str = Field(...)
    asset_path: str = Field(...)
    data: Optional[bytes] = Field(None)
    version: str = Field("v0.00")
    tags: List[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Returns a JSON-like string representation of the artifact."""
        return (
            f'{{"type": "{self.type}", "name": "{self.name}", '
            f'"asset_path": "{self.asset_path}", "version": "{self.version}"}}'
        )

    def id(self) -> str:
        """
        Generates a unique ID for the artifact.

        Returns:
            str: Unique identifier combining asset path and version.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return encoded_path + self.version

    def read(self) -> Optional[bytes]:
        """
        Returns the binary data of the artifact.

        Returns:
            Optional[bytes]: Binary data of the artifact,
            or None if not available.
        """
        return self.data

    def save(self, binary_data: bytes) -> None:
        """
        Saves binary data to the artifact.

        Args:
            binary_data (bytes): Data to store in the artifact.
        """
        self.data = binary_data
