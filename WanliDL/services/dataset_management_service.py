from WanliDL.services import DatabaseService, IDatabaseRecordManagement, IFileManagement
from WanliDL.services import ConfigurationService
config_service = ConfigurationService()

import os
import uuid
import json
from datetime import datetime
import shutil

class DatasetManagementService(DatabaseService, IDatabaseRecordManagement, IFileManagement):
    def __init__(self, db_path='./database/db.sqlite'):
        super().__init__(db_path)
        self.clean_orphan_files(config_service.get_dataset_dir())

    def ensure_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_records';")
        if not cursor.fetchone():
            create_table_query = '''
            CREATE TABLE dataset_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                split TEXT NOT NULL,
                total_split INTEGER NOT NULL,
                ratio TEXT NOT NULL,
                path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                random_seed INTEGER,
                mode TEXT,
                properties TEXT
            );'''
            cursor.execute(create_table_query)
            self.conn.commit()
        cursor.close()

    def add_record(self, name, splits, ratios, file_paths, output_folder, random_seed=None, mode=None, properties=None):
        if len(splits) != len(ratios) or len(file_paths) != len(splits):
            raise ValueError("The lengths of splits, ratios, and file_paths must be equal.")
        
        if sum(ratios) != 100:
            raise ValueError("The sum of ratios must be 100.")

        cursor = self.conn.cursor()
        for split, ratio, file_path in zip(splits, ratios, file_paths):
            random_filename = str(uuid.uuid4()) + os.path.splitext(file_path)[1]
            new_path = os.path.join(output_folder, random_filename)
            shutil.copy(file_path, new_path)
            os.remove(file_path)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_query = '''INSERT INTO dataset_records
                              (name, split, total_split, ratio, path, timestamp, random_seed, mode, properties)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);'''
            cursor.execute(insert_query, (name, split, len(splits), ratio, new_path, timestamp, random_seed, mode, json.dumps(properties) if properties else None))
        self.conn.commit()
        cursor.close()

    def find_records(self, name, total_splits, random_seed=None):
        cursor = self.conn.cursor()
        query = '''
        SELECT * FROM dataset_records
        WHERE name = ? AND total_split = ? {random_seed_clause}
        ORDER BY timestamp DESC;
        '''
        params = [name, total_splits]
        if random_seed is not None:
            random_seed_clause = "AND random_seed = ?"
            params.append(random_seed)
        else:
            random_seed_clause = "AND random_seed IS NULL"
        
        cursor.execute(query.format(random_seed_clause=random_seed_clause), tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def delete_record(self, name, total_splits, random_seed=None):
        cursor = self.conn.cursor()
        select_query = '''
        SELECT path FROM dataset_records
        WHERE name = ? AND total_split = ? {random_seed_clause}
        '''
        delete_query = '''
        DELETE FROM dataset_records
        WHERE name = ? AND total_split = ? {random_seed_clause}
        '''
        params = [name, total_splits]
        if random_seed is not None:
            random_seed_clause = "AND random_seed = ?"
            params.append(random_seed)
        else:
            random_seed_clause = "AND random_seed IS NULL"

        cursor.execute(select_query.format(random_seed_clause=random_seed_clause), tuple(params))
        rows = cursor.fetchall()
        for row in rows:
            if os.path.exists(row[0]):
                os.remove(row[0])

        cursor.execute(delete_query.format(random_seed_clause=random_seed_clause), tuple(params))
        self.conn.commit()
        cursor.close()

    def verify_and_clean_files(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, path FROM dataset_records;")
        records = cursor.fetchall()
        ids_to_delete = [record[0] for record in records if not os.path.exists(record[1])]

        for record_id in ids_to_delete:
            cursor.execute("DELETE FROM dataset_records WHERE id = ?;", (record_id,))
        
        self.conn.commit()
        cursor.close()

    def clean_orphan_files(self, directory):
        cursor = self.conn.cursor()
        cursor.execute("SELECT path FROM dataset_records;")
        valid_paths = set(record[0] for record in cursor.fetchall())
        
        all_files = {os.path.join(directory, file) for file in os.listdir(directory)}
        orphan_files = all_files - valid_paths

        for file_path in orphan_files:
            os.remove(file_path)

        cursor.close()


def create_test_dataset_management_service():

    # Initialize the DatasetManagementService with an in-memory SQLite database
    test_service = DatasetManagementService(db_path=':memory:')

    return test_service