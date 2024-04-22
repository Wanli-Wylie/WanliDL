from . import IFeatureStore
import sqlite3
from datetime import datetime
import json
from typing import Dict, Optional

class FasterRCNNFeatureStore(IFeatureStore):

    def __init__(self, db_path):
        self.db_path = db_path
        self.connect()

    def connect(self):
        """Establishes a connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.ensure_feature_table_exists()

    def ensure_feature_table_exists(self):
        """Ensures that the feature table exists in the database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY,
                shape TEXT,
                file_path TEXT,
                num_objects_detected INT,
                feature_path TEXT
            );
        ''')
        self.conn.commit()
        cursor.close()

    def close_connection(self):
        """Closes the connection to the database."""
        if self.conn:
            self.conn.close()

    def add_feature_record(self, shape: str, file_path: str, num_objects_detected: int, feature_path: str):
        """
        Adds a new record of features to the database.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO features (shape, file_path, num_objects_detected, feature_path)
            VALUES (?, ?, ?, ?);
        ''', (shape, file_path, num_objects_detected, feature_path))
        self.conn.commit()
        cursor.close()

    def get_feature_record(self, record_id: int) -> Optional[Dict[str, any]]:
        """
        Retrieves a feature record from the database by its identifier.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM features WHERE id = ?;
        ''', (record_id,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            return {
                'id': row[0],
                'shape': row[1],
                'file_path': row[2],
                'num_objects_detected': row[3],
                'feature_path': row[4]
            }
        return None

    def update_feature_record(self, record_id: int, shape: str, file_path: str, num_objects_detected: int, feature_path: str):
        """
        Updates an existing feature record in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE features
            SET shape = ?, file_path = ?, num_objects_detected = ?, feature_path = ?
            WHERE id = ?;
        ''', (shape, file_path, num_objects_detected, feature_path, record_id))
        self.conn.commit()
        cursor.close()

    def delete_feature_record(self, record_id: int):
        """
        Deletes a feature record from the database by its identifier.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM features WHERE id = ?;
        ''', (record_id,))
        self.conn.commit()
        cursor.close()

    def list_feature_records(self) -> Optional[Dict[str, any]]:
        """
        Lists all feature records in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM features;')
        rows = cursor.fetchall()
        cursor.close()
        return [{'id': row[0], 'shape': row[1], 'file_path': row[2], 'num_objects_detected': row[3], 'feature_path': row[4]} for row in rows]
    
    def record_exists(self, file_path: str) -> bool:
        """
        Checks if a feature record already exists for the given file path.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1 FROM features WHERE file_path = ?', (file_path,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists