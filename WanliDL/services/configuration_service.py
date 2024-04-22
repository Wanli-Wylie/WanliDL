import yaml

class ConfigurationService:
    def __init__(self, config_path='./configs/registry.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_tokenizer_path(self, model_name):
        return self.config["models"][model_name]['path']

    def get_model_config(self):
        return self.config["models"]
    
    def get_dataset_config(self):
        return self.config["datasets"]
    
    def get_temp_dir(self):
        return self.config["temp_dir"]
    
    def get_log_dir(self):
        return self.config["log_dir"]
    
    def get_log_interval(self):
        return self.config["log_interval"]
    
    def get_weights_dir(self):
        return self.config["weights_dir"]
    
    def get_dataset_dir(self):
        return self.config["dataset_dir"]
    
    def get_output_dir(self):
        return self.config["output_dir"]
