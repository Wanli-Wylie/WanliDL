import os
import uuid
from transformers import AutoTokenizer, AutoModel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

from VSRN import VSRN
from data_loader_factory import VSRNDataLoaderFactory

import sys
sys.path.append('.')
from WanliDL.engines import BaseTrainer
from WanliDL.services import ModelCheckpointService, ConfigurationService
from WanliDL.utils import OptimizerFactory, SchedulerFactory

# model_checkpoint_service = ModelCheckpointService()
config_service = ConfigurationService()

class VSRNTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, train_loader, logger, config, checkpoint_service=None):

        self.model = model.to(config.get('device', 'cuda'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.logger = logger
        self.config = config
        self.checkpoint_service = checkpoint_service
        self.weights_dir = config_service.get_weights_dir()
        os.makedirs(self.weights_dir, exist_ok=True)
        self.device = self.config.get('device', 'cuda')

    def step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        region_features = batch['region_features'].to(self.device)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        vision_feature = self.model(region_features)
        total_loss = 0
        
        # Compute losses for each of the 5 captions
        for i in range(5):
            text_feature_input_ids = input_ids[i].to(self.device)
            text_feature_attention_mask = attention_mask[i].to(self.device)
            
            matching_loss = self.model.compute_matching_loss(vision_feature, text_feature_input_ids)
            generation_loss = self.model.compute_generation_loss(vision_feature, text_feature_input_ids, text_feature_attention_mask)

            total_loss += matching_loss + generation_loss

        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()  
        
        return total_loss.item()

    def run(self):
        trainer = Engine(self.step)
        training_datasets = str(self.config['training_datasets']) if isinstance(self.config['training_datasets'], int) else self.config['training_datasets']
        model_name = self.config['model_name']
        random_seed = self.config['random_seed']
        
        checkpointer = ModelCheckpoint(
            dirname=self.weights_dir,
            filename_prefix=training_datasets,
            n_saved=None,  
            create_dir=True,
            require_empty=False
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': self.model, 'optimizer': self.optimizer})

        @trainer.on(Events.EPOCH_COMPLETED)
        def external_checkpoint_handler(engine):
            iteration = engine.state.iteration
            src_name = f"{training_datasets}_checkpoint_{iteration}.pt"
            dst_name = f"{uuid.uuid4()}.pt"
            self.checkpoint_service.move_and_record_checkpoint(
                src_name, 
                dst_name, 
                model_name, 
                training_datasets, 
                random_seed, 
                iteration)

        # Optional: Attach progress bar and metrics
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer)

        log_interval = self.config.get('log_interval', 100)
        @trainer.on(Events.ITERATION_COMPLETED(event_filter=lambda _, i: i % log_interval == 0))
        def log_training_loss(engine):
            self.logger.info(f"Iteration {engine.state.iteration} - Loss: {engine.state.output:.4f}")

        trainer.run(self.train_loader, max_epochs=self.config['epoches'])
        self.logger.info("Training completed")

if __name__ == "__main__":
    model = VSRN(D=2048, bert=AutoModel.from_pretrained('./pretrained/bert-base-uncased'))
    dataloader = VSRNDataLoaderFactory(
        'datasets/vision/MSCOCO/train2014_features.sqlite', 
        AutoTokenizer.from_pretrained('pretrained/bert-base-uncased'), 
        './datasets/vision/MSCOCO/train2014_captions.json').create(1024)
    
    optimizer_config = {
        'OPTIMIZER_NAME': 'Adam',
        'BASE_LR': 0.0002,
        'WEIGHT_DECAY': 0.0,  # Assuming no weight decay unless specified otherwise
        'BIAS_LR_FACTOR': 1,  # No special treatment for biases
        'WEIGHT_DECAY_BIAS': 0.0  # Assuming no separate weight decay for biases
    }

    scheduler_config = {
        'name': 'StepLR',
        'step_size': 15,  # Adjust the learning rate every 15 epochs
        'gamma': 0.1      # Reduce the learning rate to one-tenth of its previous value
    }

    optimizer = OptimizerFactory().create(optimizer_config, model.parameters())
    scheduler = SchedulerFactory().create(scheduler_config, optimizer)
