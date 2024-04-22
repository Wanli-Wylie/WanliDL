

from abc import ABC, abstractmethod
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

class IFeatureStore(ABC):
    """
    Interface for a Feature Store service that manages pre-computed features for machine learning models.
    """

    @abstractmethod
    def connect(self):
        """
        Establishes a connection to the SQLite database.
        """
        pass

    @abstractmethod
    def ensure_feature_table_exists(self):
        """
        Ensures that the table for storing features exists in the database.
        """
        pass

    @abstractmethod
    def add_feature_record(self, feature_id: str, features: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a new record of features to the database.

        :param feature_id: Unique identifier for the feature record.
        :param features: Dictionary containing the features.
        :param metadata: Optional dictionary containing additional metadata about the features.
        """
        pass

    @abstractmethod
    def get_feature_record(self, feature_id: str) -> Dict[str, Any]:
        """
        Retrieves a feature record from the database by its identifier.

        :param feature_id: The unique identifier for the feature record.
        :return: Dictionary containing the features and metadata.
        """
        pass

    @abstractmethod
    def delete_feature_record(self, feature_id: str):
        """
        Deletes a feature record from the database by its identifier.

        :param feature_id: The unique identifier for the feature record to be deleted.
        """
        pass

    @abstractmethod
    def list_feature_records(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Lists all feature records, optionally filtered by specified criteria.

        :param filter_criteria: Optional dictionary for filtering records based on certain conditions.
        :return: List of dictionaries containing the features and metadata of each record.
        """
        pass

    @abstractmethod
    def close_connection(self):
        """
        Closes the connection to the database.
        """
        pass

from .faster_r_cnn import FasterRCNNFeatureStore
from .bert import BertFeatureStore