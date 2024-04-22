import argparse
import yaml
import os
import sys
import json

import torch.nn.functional as F

# Make sure the current directory is in the Python path
sys.path.append('.')

# Import necessary modules from the local package
from scripts.data.jsonl_data_loader_factory import JSONLDataLoaderFactory
from scripts.services.logging_service import LoggingService
from scripts.utils import ModelFactory, OptimizerFactory, SchedulerFactory, LossFunctionFactory

model_factory = ModelFactory()
data_loader_factory = JSONLDataLoaderFactory()
optimizer_factory = OptimizerFactory()
scheduler_factory = SchedulerFactory()
logging_service = LoggingService()
loss_function_factory = LossFunctionFactory()

from scripts.engines.trainer import Trainer  # Import the Trainer class

def make_data_loader(config):
    return data_loader_factory.make_data_loader(
        mode=config.get('mode'),
        dataset_name=config.get('datasets'),
        random_seed=config.get("random_seed"),
        tokenizer_name=config.get("tokenizer_name"),
        batch_size=config['batch_size'],
        shuffle=config.get('shuffle', False),
        num_workers=config.get('num_workers', 0)
    )

def train(config):

    # Unpack model settings from the config file
    model = model_factory.create(config["model"])

    # Create the data loader for training
    dataloader = make_data_loader(config['data'])

    # Unpack solver settings from the config file
    optimizer = optimizer_factory.create(config['optimizer'], model.parameters())

    # Unpack LR scheduler settings from the config file
    lr_scheduler = scheduler_factory.create(config['scheduler'], optimizer)

    # Get loss function
    loss_function = loss_function_factory.create(config['loss_function'])

    # Set up the logging service
    _, logger = logging_service.setup_logger(config['logging'], json.dumps(config))

    # Set up the trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss_function,
        train_loader=dataloader,
        logger=logger,
        config=config.get("experiment_info")
    )

    # Run the training process
    trainer.run()

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
    train(config)

if __name__ == "__main__":
    main()
