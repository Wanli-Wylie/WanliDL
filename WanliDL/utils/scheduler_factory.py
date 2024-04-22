import torch.optim as optim
from . import IFactory

class SchedulerFactory(IFactory):
    def __init__(self):
        # Define available schedulers
        self.available_schedulers = {
            'StepLR': optim.lr_scheduler.StepLR,
            'ExponentialLR': optim.lr_scheduler.ExponentialLR
        }

    def create(self, config, optimizer):
        """
        Creates a learning rate scheduler for the given optimizer based on the config dictionary.
        
        Args:
            config (dict): Configuration dictionary containing scheduler settings.
            optimizer (torch.optim.Optimizer): Optimizer instance to which the scheduler will be applied.
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: Configured scheduler instance.
        
        Raises:
            ValueError: If the specified scheduler is not supported.
        """
        scheduler_name = config.get('name')
        if scheduler_name not in self.available_schedulers:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")

        scheduler_class = self.available_schedulers[scheduler_name]
        scheduler_args = {k: v for k, v in config.items() if k in ['gamma', 'step_size']}  # Filter relevant args
        return scheduler_class(optimizer, **scheduler_args)
