from . import IFeatureStore
import sqlite3
from datetime import datetime
import json
from typing import Dict, Optional

class BertFeatureStore(IFeatureStore):

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
            CREATE TABLE IF NOT EXISTS bert_features (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                feature_path TEXT
            );
        ''')
        self.conn.commit()
        cursor.close()

    def close_connection(self):
        """Closes the connection to the database."""
        if self.conn:
            self.conn.close()

    def add_feature_record(self, file_path: str, feature_path: str):
        """
        Adds a new record of features to the database.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO bert_features (file_path, feature_path)
            VALUES (?, ?);
        ''', (file_path, feature_path))
        self.conn.commit()
        cursor.close()

    def get_feature_record(self, record_id: int) -> Optional[Dict[str, any]]:
        """
        Retrieves a feature record from the database by its identifier.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM bert_features WHERE id = ?;
        ''', (record_id,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            return {
                'id': row[0],
                'file_path': row[1],
                'feature_path': row[2]
            }
        return None

    def update_feature_record(self, record_id: int, file_path: str, feature_path: str):
        """
        Updates an existing feature record in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE bert_features
            SET file_path = ?, feature_path = ?
            WHERE id = ?;
        ''', (file_path, feature_path, record_id))
        self.conn.commit()
        cursor.close()

    def delete_feature_record(self, record_id: int):
        """
        Deletes a feature record from the database by its identifier.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM bert_features WHERE id = ?;
        ''', (record_id,))
        self.conn.commit()
        cursor.close()

    def list_feature_records(self) -> Optional[Dict[str, any]]:
        """
        Lists all feature records in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM bert_features;')
        rows = cursor.fetchall()
        cursor.close()
        return [{'id': row[0], 'file_path': row[1], 'feature_path': row[2]} for row in rows]
    
    def record_exists(self, file_path: str) -> bool:
        """
        Checks if a feature record already exists for the given file path.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1 FROM bert_features WHERE file_path = ?', (file_path,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists
