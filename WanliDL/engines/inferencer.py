import torch
import os
import torch
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
import sys
import json
import uuid

sys.path.append('.')

from scripts.engines import BaseInferencer
from scripts.services import ModelOutputManagementService, ConfigurationService

model_output_management_service = ModelOutputManagementService()
config_service = ConfigurationService()

class Inferencer(BaseInferencer):
    def __init__(self, model, weights_id, logger, config):
        """
        Initializes the inference engine with a pre-loaded model, logger, and management service.
        :param model: A PyTorch model already instantiated and loaded with weights.
        :param weights_id: ID of the model's weights file.
        :param logger: Logger instance for logging information.
        :param config: A configuration dictionary containing model and other settings.
        :param management_service: Instance of ModelOutputManagementService for managing outputs.
        """
        self.model = model
        self.weights_id = weights_id
        self.logger = logger
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Ensure the model is in evaluation mode

    def run(self, data_iterable, dataset_id):
        """
        Run inference over an iterable of data and log outputs to the management service.
        :param data_iterable: Iterable that yields data.
        :param output_path: Path to save inference results.
        :return: List of inference results.
        """
        results = []

        def step(engine, batch):
            """
            Perform a single inference step.
            :param engine: Ignite Engine instance.
            :param batch: Batch of data to perform inference on.
            :return: Inference results.
            """
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                output = self.model(**inputs)
                result = [{'output': o, 
                           'labels': l} for o, l in zip(
                    output.cpu().numpy().tolist(), 
                    batch['labels'].cpu().numpy().tolist())]
                results.extend(result)
                return output

        engine = Engine(step)

        pbar = ProgressBar(persist=False)
        pbar.attach(engine)

        engine.run(data_iterable)

        # Save the output and insert a record into the management database
        file_path = self.save_results(results)
        model_output_management_service.add_record(
            model_name=self.config['model_name'],
            weights_id=self.weights_id,
            input_dataset=dataset_id,
            random_seed=self.config.get('random_seed'),
            output_path=file_path)
        return results

    def save_results(self, results):
        """
        Save inference results to a specified path.
        :param results: Inference results to save.
        :param output_path: Path to save the results.
        """

        file_path = os.path.join(config_service.get_output_dir(), f"{uuid.uuid4()}.jsonl")

        with open(file_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        return file_path
