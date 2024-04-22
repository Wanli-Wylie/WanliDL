import sqlite3
import json
from datetime import datetime

from WanliDL.services import DatabaseService, IDatabaseRecordManagement, ConfigurationService

config_service = ConfigurationService()

class ModelOutputManagementService(DatabaseService, IDatabaseRecordManagement):
    def __init__(self, db_path='./database/db.sqlite'):
        super().__init__(db_path)

    def ensure_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_output';")
        if not cursor.fetchone():
            create_table_query = '''
            CREATE TABLE model_output (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                weights_id INTEGER NOT NULL,
                input_dataset INTEGER NOT NULL,
                random_seed INTEGER,
                output_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                properties TEXT,
                FOREIGN KEY (weights_id) REFERENCES model_checkpoint(id),
                FOREIGN KEY (input_dataset) REFERENCES dataset_records(id)
            );'''
            cursor.execute(create_table_query)
            self.conn.commit()
        cursor.close()

    def add_record(self, model_name, weights_id, input_dataset, random_seed, output_path, properties=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        insert_query = '''INSERT INTO model_output
                          (model_name, weights_id, input_dataset, random_seed, output_path, timestamp, properties)
                          VALUES (?, ?, ?, ?, ?, ?, ?);'''
        cursor.execute(insert_query, (model_name, weights_id, input_dataset, random_seed, output_path, timestamp, json.dumps(properties) if properties else None))
        self.conn.commit()
        cursor.close()

    def find_records(self, model_name):
        cursor = self.conn.cursor()
        query = '''
        SELECT * FROM model_output WHERE model_name = ?;
        '''
        cursor.execute(query, (model_name,))
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def delete_record(self, id):
        cursor = self.conn.cursor()
        delete_query = '''DELETE FROM model_output WHERE id = ?;'''
        cursor.execute(delete_query, (id,))
        self.conn.commit()
        cursor.close()
