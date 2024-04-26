from abc import ABC, abstractmethod

# Interface for data loader management
class IDatasetOperations(ABC):

    @abstractmethod
    def search_dataset(self, mode, dataset_name, random_seed):
        pass

    @abstractmethod
    def build_dataset_from_path(self, dataset_path, tokenizer_name):
        pass

    @abstractmethod
    def make_data_loader(self, config):
        pass


# Interface for data loader factory
class IDataLoaderFactory(ABC):

    @abstractmethod
    def search_dataset(self, mode, dataset_name, random_seed):
        pass

    @abstractmethod
    def build_dataset_from_path(self, dataset_path, tokenizer_name):
        pass

    @abstractmethod
    def make_data_loader(self, config):
        pass

