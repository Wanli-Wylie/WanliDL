import argparse
import yaml
import sys
import json

import torch.nn.functional as F

# Make sure the current directory is in the Python path
sys.path.append('.')

# Import necessary modules from the local package
from scripts.data.jsonl_data_loader_factory import JSONLDataLoaderFactory
from scripts.services import LoggingService, ModelCheckpointService, ModelOutputManagementService
from scripts.utils import ModelFactory
from scripts.engines.inferencer import Inferencer  # Import the Trainer class

model_service = ModelFactory()
data_loader_factory = JSONLDataLoaderFactory()
logging_service = LoggingService()
model_checkpoint_service = ModelCheckpointService()
model_output_management_service = ModelOutputManagementService()

def find_latest_weights(config_weights):

    model_name = config_weights['model_name']
    training_datasets = config_weights['training_datasets']
    random_seed = config_weights['random_seed']

    weights_id, _, _, _, weights_path, _, _, _ = model_checkpoint_service.find_records(
        model_name,
        training_datasets,
        random_seed)[0]
    
    return weights_id, weights_path

def make_data_loader(config):

    dataset_id = data_loader_factory.search_dataset(
        mode=config.get('mode'),
        dataset_name=config.get('datasets'),
        random_seed=config.get("random_seed"),
    )[0][0]

    data_loader = data_loader_factory.make_data_loader(
        mode=config.get('mode'),
        dataset_name=config.get('datasets'),
        random_seed=config.get("random_seed"),
        tokenizer_name=config.get("tokenizer_name"),
        batch_size=config['batch_size'],
        shuffle=config.get('shuffle', False),
        num_workers=config.get('num_workers', 0)
    )

    return dataset_id, data_loader

def inference(config):

    weights_id, weights_path = find_latest_weights(config["weights"])

    model = model_service.create(config["model"], weights_path)

    dataset_id, data_loader = make_data_loader(config['data'])

    # Set up the logging service
    _, logger = logging_service.setup_logger(config['logging'], json.dumps(config))

    # Set up the trainer
    inferencer = Inferencer(
        model=model,
        weights_id=weights_id,
        logger=logger,
        config=config.get("experiment_info"))

    # Run the training process
    inferencer.run(data_loader, dataset_id)

    # Close the logging service if needed
    logging_service.close()

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train a model based on a configuration file")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Call the train function with the loaded configuration
    inference(config)

if __name__ == "__main__":
    main()
