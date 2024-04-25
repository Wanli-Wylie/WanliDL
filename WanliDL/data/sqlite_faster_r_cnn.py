import sqlite3
import json
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

class FasterRCNNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the feature file path
        feature_paths = [item[4] for item in self.data[idx]]

        boxes = []
        labels = []
        scores = []
        
        # Load feature from file (assuming it's a NumPy file for simplicity)
        for feature_path in feature_paths:
            with open(feature_path, 'rb') as file:
                feature = json.load(file)
                boxes.append(torch.tensor(feature['boxes']).unsqueeze(0))
                labels.append(torch.tensor(feature['labels']).unsqueeze(0))
                scores.append(torch.tensor(feature['scores']).unsqueeze(0))
        
        return {
            'boxes': torch.cat(boxes, dim=0),
            'labels': torch.cat(labels, dim=0),
            'scores': torch.cat(scores, dim=0)
        }
        
        # Reshape tensor to match expected size (1, B, k, D)
        # We assume the tensor loaded is (k, D)
        # return feature_tensor.unsqueeze(0)

class FasterRCNNDataLoaderFactory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.data = self._load_data()

    def _load_data(self):
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Execute query to fetch all data
        cursor.execute("SELECT * FROM features")
        data = cursor.fetchall()
        conn.close()
        
        # Group data by 'num_objects_detected'
        grouped_data = defaultdict(list)
        for row in data:
            # Each row: (id, shape, file_path, num_objects_detected, feature_path)
            grouped_data[row[3]].append(row)
        
        return grouped_data

    def create(self, batch_size):
        # Create a dictionary of dataloaders for different 'num_objects_detected'
        data = []
        for _, grouped_items in self.data.items():
            for ind in range(batch_size, len(grouped_items) + 1, batch_size):
                batch = grouped_items[ind-batch_size:ind]
                data.append(batch)
            # print(f"Creating DataLoader for 'num_objects_detected'={k}, with {len(grouped_items)} items")
        dataset = FasterRCNNDataset(data)
        dataloader = DataLoader(dataset, shuffle=True)
        return dataloader

# Usage
db_path = 'datasets/vision/Flickr30k/features.sqlite'
factory = FasterRCNNDataLoaderFactory(db_path)
batch_size = 64
dataloaders = factory.create(batch_size)
for i, data in enumerate(dataloaders):

    print(f"Batch {i+1}: {data['boxes'].shape}")
    if i == 2:
        break