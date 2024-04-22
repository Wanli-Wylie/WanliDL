import sqlite3
from abc import ABC, abstractmethod

# Abstract base class for database interaction
class DatabaseService(ABC):
    def __init__(self, db_path='./database/db.sqlite'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.ensure_table_exists()

    @abstractmethod
    def ensure_table_exists(self):
        pass

    def close_connection(self):
        self.conn.close()

    def __del__(self):
        self.close_connection()

# Interface for record management in the database
class IDatabaseRecordManagement(ABC):
    @abstractmethod
    def add_record(self, *args, **kwargs):
        pass

    @abstractmethod
    def find_records(self, *args, **kwargs):
        pass

    @abstractmethod
    def delete_record(self, *args, **kwargs):
        pass

# Interface for file management
class IFileManagement(ABC):
    @abstractmethod
    def verify_and_clean_files(self):
        pass

    @abstractmethod
    def clean_orphan_files(self, directory):
        pass

# Interface for data loader management
class IDatasetOperations(ABC):

    @abstractmethod
    def search_dataset(self, mode, dataset_name, random_seed):
        pass

    @abstractmethod
    def build_dataset_from_path(self, dataset_path, tokenizer_name):
        pass

    @abstractmethod
    def make_data_loader(self, config):
        pass



from .configuration_service import ConfigurationService
from .dataset_management_service import DatasetManagementService
from .logging_service import LoggingService
from .model_checkpoint import ModelCheckpointService
from .model_output_management_service import ModelOutputManagementService