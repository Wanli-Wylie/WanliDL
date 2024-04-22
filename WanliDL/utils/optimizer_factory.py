import torch.optim as optim
from . import IFactory

class OptimizerFactory(IFactory):
    def __init__(self):
        # Define available optimizers in a dictionary
        self.available_optimizers = {
            'SGD': optim.SGD,
            'Adam': optim.Adam,
            'RMSprop': optim.RMSprop
        }

    def create(self, config, parameters):
        """
        Creates an optimizer for the given parameters based on the config dictionary.
        
        Args:
            config (dict): Configuration dictionary containing optimizer settings.
            parameters (iterable): An iterable of parameters to optimize (usually model.parameters()).
        
        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        
        Raises:
            ValueError: If the specified optimizer is not supported or required keys are missing.
        """
        optimizer_name = config.get('OPTIMIZER_NAME')
        if optimizer_name not in self.available_optimizers:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

        # Create parameter groups
        base_lr = config.get('BASE_LR')
        weight_decay = config.get('WEIGHT_DECAY')
        momentum = config.get('MOMENTUM', 0)  # Default momentum to 0 if not specified
        
        # Custom handling for biases (optional)
        bias_lr = base_lr * config.get('BIAS_LR_FACTOR', 1)
        weight_decay_bias = config.get('WEIGHT_DECAY_BIAS', weight_decay)

        param_groups = [
            {'params': [p for p in parameters if p.requires_grad and p.ndim != 1], 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': [p for p in parameters if p.requires_grad and p.ndim == 1], 'lr': bias_lr, 'weight_decay': weight_decay_bias}
        ]

        return self.available_optimizers[optimizer_name](param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
