import torch
import torch.nn as nn
from scripts.services import ConfigurationService
from . import IFactory
import sys
sys.path.append('.')
from scripts.model.nlp import BertForBinaryClassification

class ModelFactory(IFactory):
    def __init__(self):
        self.config_service = ConfigurationService()
        self.model_list = self.config_service.get_model_config()

    def create(self, config, weights=None):
        name = config.get("name")
        type = config.get("type")

        if name not in self.model_list:
            raise ValueError(f"Model name {name} not found in the configuration file")
        if type == "bert_for_binary_classification":
            model_path = self.model_list[name].get("path")
            model = BertForBinaryClassification(model_path)
        
        if weights:
            model.load_state_dict(torch.load(weights).get("model"))

        return model

    def load_weights(self, model, weights_path):
        model.load_state_dict(torch.load(weights_path).get("model"))
        return model