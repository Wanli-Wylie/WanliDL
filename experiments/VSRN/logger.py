import logging
import os
import sys
import uuid
import json

sys.path.append(".")

from WanliDL.services.configuration_service import ConfigurationService

class LoggingService:
    def __init__(self):
        # Create a configuration service instance using the provided config
        self.config_service = ConfigurationService()
        
        # Initialize logger manager
        self.loggers = {}

    def setup_logger(self, config):
        # Retrieve the directory to save logs from configuration
        save_dir = self.config_service.get_log_dir()
        
        # Setup logger basic configuration
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

            # Optionally save log metadata or settings to a JSON file
            self.save_log_config(config, log_file_path)

        self.loggers[experiment_id] = logger
        return (experiment_id, logger)

    def save_log_config(self, config, file_path):
        # Save logger configuration to a JSON file
        config_file_path = file_path + '.json'
        with open(config_file_path, 'w') as file:
            json.dump(config, file, indent=4)

    def get_logger(self, name):
        # Retrieve a logger by name
        return self.loggers.get(name, None)
