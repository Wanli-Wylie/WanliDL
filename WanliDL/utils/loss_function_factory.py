import torch.nn.functional as F
from . import IFactory

class LossFunctionFactory(IFactory):
    def __init__(self):
        # Initialize a dictionary to map configuration strings to loss functions
        self.loss_functions = {
            'binary_cross_entropy': F.binary_cross_entropy
        }

    def create(self, config):
        """
        Retrieves the loss function based on the provided config string.
        
        Args:
            config (str): The configuration string identifying the loss function.
        
        Returns:
            function: The loss function corresponding to the config.
        
        Raises:
            ValueError: If the specified loss function is not supported.
        """
        if config not in self.loss_functions:
            raise ValueError(f"Loss function '{config}' is not supported.")
        return self.loss_functions[config]
