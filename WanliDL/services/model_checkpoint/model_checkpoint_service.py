import sqlite3
import os
from datetime import datetime

from WanliDL.services import DatabaseService, IDatabaseRecordManagement, IFileManagement, ConfigurationService

config_service = ConfigurationService()

class ModelCheckpointService(DatabaseService, IDatabaseRecordManagement, IFileManagement):
    def __init__(self, db_path='./database/db.sqlite', n_saved=5):
        super().__init__(db_path)
        self.n_saved = n_saved
        self.ensure_table_exists()
        self.verify_and_clean_files()
        self.clean_orphan_files(config_service.get_weights_dir())

    def ensure_table_exists(self):
        cursor = self.conn.cursor()
        # Check and create the model_checkpoint table if necessary
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_checkpoint';")
        if not cursor.fetchone():
            create_table_query = '''
            CREATE TABLE model_checkpoint (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                training_datasets TEXT NOT NULL,
                random_seed INT NOT NULL,
                weights_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                properties TEXT
            );'''
            cursor.execute(create_table_query)
            self.conn.commit()
        cursor.close()

    def add_record(self, model_name, dataset_ids, random_seed, weights_path, iterations, properties=None):

        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]

        training_datasets = "+".join(map(str, dataset_ids))  # Joining dataset IDs into a single string
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        insert_query = '''
        INSERT INTO model_checkpoint (
            model_name, 
            training_datasets, 
            random_seed, 
            weights_path, 
            timestamp, 
            iterations,
            properties
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        '''
        cursor.execute(insert_query, (model_name, training_datasets, random_seed, weights_path, timestamp, iterations, properties))
        self.conn.commit()
        cursor.close()

    def find_records(self, model_name, dataset_ids, random_seed):

        training_datasets = str(dataset_ids) if isinstance(dataset_ids, int) else dataset_ids
        cursor = self.conn.cursor()
        query = '''
        SELECT * FROM model_checkpoint
        WHERE model_name=? AND training_datasets=? AND random_seed=?
        ORDER BY iterations DESC;
        '''
        cursor.execute(query, (model_name, training_datasets, random_seed))
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def delete_record(self, id):
        cursor = self.conn.cursor()
        delete_query = "DELETE FROM model_checkpoint WHERE id=?"
        cursor.execute(delete_query, (id,))
        self.conn.commit()
        cursor.close()

    def verify_and_clean_files(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, weights_path FROM model_checkpoint")
        records = cursor.fetchall()
        for record in records:
            id, path = record
            if not os.path.exists(path):
                self.delete_record(id)
        self.conn.commit()
        cursor.close()

    def clean_orphan_files(self, directory):
        cursor = self.conn.cursor()
        cursor.execute("SELECT weights_path FROM model_checkpoint")
        valid_paths = set(record[0] for record in cursor.fetchall())
        all_files = {os.path.join(directory, file) for file in os.listdir(directory)}
        orphan_files = all_files - valid_paths
        for file_path in orphan_files:
            os.remove(file_path)
        cursor.close()

    def move_and_record_checkpoint(self, src_name, dst_name, model_name, training_datasets, random_seed, iteration):
        weights_dir = config_service.get_weights_dir()
        src_path = os.path.join(weights_dir, src_name)
        dst_path = os.path.join(weights_dir, dst_name)
        os.rename(src_path, dst_path)
        self.add_record(model_name, training_datasets, random_seed, dst_path, iteration)