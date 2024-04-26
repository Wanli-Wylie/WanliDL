import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.append('.')
# Import database management functionalities
from WanliDL.services.feature_store import FasterRCNNFeatureStore
from WanliDL.data import FasterRCNNDataLoaderFactory

class VSRNDataLoaderFactory(DataLoader):
    def __init__(self, db_path, tokenizer, caption_path):
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.caption_path = caption_path
        self.factory = FasterRCNNDataLoaderFactory(db_path, tokenizer, caption_path)
    
    def create(self, batch_size):
        return self.factory.create(batch_size)

if __name__ == '__main__':
    db_path = 'datasets/vision/MSCOCO/train2014_features.sqlite'
    tokenizer = AutoTokenizer.from_pretrained('pretrained/bert-base-uncased')
    caption_path = './datasets/vision/MSCOCO/train2014_captions.json'

    factory = FasterRCNNDataLoaderFactory(db_path, tokenizer, caption_path)

    batch_size = 8

    dataloader = VSRNDataLoaderFactory(db_path, tokenizer, caption_path).create(batch_size)

    for batch in dataloader:
        # Save the batch to a file

        batch_json = {}

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_json[k] = v.cpu().numpy().tolist()
            else:
                if isinstance(v[0], torch.Tensor):
                    batch_json[k] = [x.cpu().numpy().tolist() for x in v]
                else:
                    batch_json[k] = v

        with open('batch.json', 'w') as f:
            json.dump(batch_json, f)
        break