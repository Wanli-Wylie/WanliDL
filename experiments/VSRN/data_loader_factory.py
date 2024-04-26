import torch
import json
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
# Import database management functionalities
from WanliDL.services.feature_store import FasterRCNNFeatureStore
from WanliDL.data import FasterRCNNDataLoaderFactory

factory = FasterRCNNDataLoaderFactory(db_path='./datasets/vision/MSCOCO/train2014_features.sqlite')

class VSRNDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn):
        super(VSRNDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn
        )

if __name__ == '__main__':
    pass