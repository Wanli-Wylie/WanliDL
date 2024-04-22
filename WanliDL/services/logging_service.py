import logging
import os
import sqlite3
import sys
import uuid
import json

import sys
sys.path.append(".")

from WanliDL.services.configuration_service import ConfigurationService

config_service = ConfigurationService()

class LoggingService:
    def __init__(self, db_path="database/db.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.loggers = {}  # To manage multiple loggers
        self.init_db()
        self.verify_and_clean_logs()

    def init_db(self):
        # Initialize the database to store logs information with new schema
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                config JSON NOT NULL,
                file_path TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def verify_and_clean_logs(self):
        # Retrieve all log records
        self.cursor.execute("SELECT id, file_path FROM logs")
        records = self.cursor.fetchall()
        
        # Check each file and delete the record if the file does not exist
        for record in records:
            log_id, file_path = record
            if not os.path.exists(file_path):
                self.cursor.execute("DELETE FROM logs WHERE id = ?", (log_id,))
        
        self.conn.commit()

    def setup_logger(self, config, experiment_config_json):

        save_dir = config_service.get_log_dir()
        experiment_id = config.get('experiment_id', 'default_id')
        mode = config.get('mode', 'train')  # Default mode is 'train'
        
        if save_dir is None:
            raise ValueError("save_dir field is required in the config")

        distributed_rank = config.get('distributed_rank', 0)

        logger = logging.getLogger(experiment_id)
        logger.setLevel(logging.DEBUG)

        if distributed_rank > 0:
            self.loggers[experiment_id] = logger
            return (experiment_id, logger)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            log_file_path = os.path.join(save_dir, f"{uuid.uuid4()}.txt")
            fh = logging.FileHandler(log_file_path, mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            # Store the log file path, experiment ID, mode, and config in the database
            config_json = json.dumps(config)
            self._save_log_info(experiment_id, mode, experiment_config_json, log_file_path)

        self.loggers[experiment_id] = logger
        return experiment_id, logger

    def _save_log_info(self, experiment_id, mode, config, file_path):
        self.cursor.execute("INSERT INTO logs (experiment_id, mode, config, file_path) VALUES (?, ?, ?, ?)",
                            (experiment_id, mode, config, file_path))
        self.conn.commit()

    def close(self):
        self.conn.close()

    def get_logger(self, name):
        return self.loggers.get(name, None)
