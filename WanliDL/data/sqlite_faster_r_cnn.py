import sqlite3
import json
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import re
from transformers import AutoTokenizer

class FasterRCNNDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the feature file path
        feature_paths = [item[4] for item in self.data[idx]]
        captions = [item[5] for item in self.data[idx]]

        num_of_regions = []
        region_features = []
        captions_indexed = {0: [], 1: [], 2: [], 3: [], 4: []}

        # Load feature from file (assuming it's a NumPy file for simplicity)
        for feature_path, caption in zip(feature_paths, captions):

            with open(feature_path, 'rb') as file:
                feature = json.load(file)
                num_of_regions.append(feature['num_of_regions'])
                region_features.append(torch.tensor(feature['region_features']).unsqueeze(0))
            
            for i in range(5):

                captions_indexed[i].append(caption[i])
        
        tokenized = {0: None, 1: None, 2: None, 3: None, 4: None}

        for i in range(5):
            tokenized[i] = self.tokenizer(captions_indexed[i], add_special_tokens=True, max_length=512, truncation=True, padding="longest", return_tensors='pt')
        
        return {
            'num_of_regions': num_of_regions,  # 'num_objects_detected
            'region_features': torch.cat(region_features, dim=0),
            'input_ids': [tokenized[i]['input_ids'].squeeze() for i in range(5)],
            'attention_mask': [tokenized[i]['attention_mask'].squeeze() for i in range(5)]
        }
        
        # Reshape tensor to match expected size (1, B, k, D)
        # We assume the tensor loaded is (k, D)
        # return feature_tensor.unsqueeze(0)

class FasterRCNNDataLoaderFactory:
    def __init__(self, db_path, tokenizer, annotation_path=None):
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.annotation_path = annotation_path
        if annotation_path:
            with open(annotation_path, 'r') as file:
                self.annotations = json.load(file)
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

            id, shape, file_path, num_objects_detected, path = row

            if self.annotation_path:
                # Search for the pattern and extract the image_id
                match = re.search(r'COCO_train2014_(\d+)\.jpg', file_path)
                if match:
                    image_id = int(match.group(1))
                    captions = self.annotations.get(str(image_id))
                else:
                    captions = None
            else:
                captions = None
            result = (id, shape, file_path, num_objects_detected, path, captions)
            grouped_data[num_objects_detected].append(result)

        return grouped_data

    def create(self, batch_size):
        # Create a dictionary of dataloaders for different 'num_objects_detected'
        data = []
        for _, grouped_items in self.data.items():
            for ind in range(batch_size, len(grouped_items) + 1, batch_size):
                batch = grouped_items[ind-batch_size:ind]
                data.append(batch)
            # print(f"Creating DataLoader for 'num_objects_detected'={k}, with {len(grouped_items)} items")
        dataset = FasterRCNNDataset(data, self.tokenizer)
        dataloader = DataLoader(dataset, shuffle=True)
        return dataloader

if __name__ == "__main__":
    # Usage
    db_path = 'datasets/vision/MSCOCO/train2014_features.sqlite'
    tokenizer = AutoTokenizer.from_pretrained('pretrained/bert-base-uncased')
    factory = FasterRCNNDataLoaderFactory(db_path, tokenizer, './datasets/vision/MSCOCO/train2014_captions.json')
    batch_size = 64
    dataloaders = factory.create(batch_size)
    for i, data in enumerate(dataloaders):
        
        print(f"Batch {i+1}: {data['input_ids'][0].shape}")
        if i == 2:
            break