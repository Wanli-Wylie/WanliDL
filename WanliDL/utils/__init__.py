from abc import ABC
import abc

class IFactory(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def create(self, *args, **kwargs):
        pass

from .model_factory import ModelFactory
from .loss_function_factory import LossFunctionFactory
from .optimizer_factory import OptimizerFactory
from .scheduler_factory import SchedulerFactory
