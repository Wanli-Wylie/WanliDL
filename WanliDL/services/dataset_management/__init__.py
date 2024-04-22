from abc import ABC, abstractmethod
import sqlite3

class DatasetManagementService(ABC):
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self.create_connection()
        self.ensure_table_exists()

    def create_connection(self):
        """Create and return a database connection."""
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

    @abstractmethod
    def ensure_table_exists(self):
        """Ensure the necessary table(s) exist in the database."""
        pass

    @abstractmethod
    def add_record(self, *args, **kwargs):
        """Add a record to the database."""
        pass

    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
