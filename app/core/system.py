from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    Manages artifact storage and retrieval.
    """
    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Args:
            database (Database): Stores artifact metadata.
            storage (Storage): Stores artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Saves artifact data and metadata.

        Args:
            artifact (Artifact): The artifact to register.
        """
        artifact_id = artifact.id() if callable(
            getattr(artifact, 'id', None)) else artifact.id
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact_id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists artifacts, optionally filtered by type.

        Args:
            type (str, optional): Type to filter by.

        Returns:
            List[Artifact]: Artifacts of the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by ID.

        Args:
            artifact_id (str): ID of the artifact.

        Returns:
            Artifact: The retrieved artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by ID.

        Args:
            artifact_id (str): ID of the artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Provides centralized access to artifact management.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Args:
            storage (LocalStorage): Storage for artifacts.
            database (Database): Stores artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Returns the singleton instance of AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Accesses the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry.
        """
        return self._registry
