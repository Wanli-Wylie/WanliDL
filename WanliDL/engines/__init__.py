from abc import ABC
import abc

class BaseTrainer(ABC):
    @abc.abstractmethod
    def step(self, engine, batch):
        """
        Perform a single step of training iteration over a batch of data.
        """
        pass
    
    @abc.abstractmethod
    def run(self):
        """
        Run the training loop.
        """
        pass

class BaseInferencer(ABC):
    
    @abc.abstractmethod
    def run(self):
        """
        Run the training loop.
        """
        pass
