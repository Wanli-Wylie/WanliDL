import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append('.')
from WanliDL.services import DatasetManagementService, ConfigurationService
from WanliDL.data.interfaces import IDatasetOperations, IDataLoaderFactory

config_service = ConfigurationService()

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = self._load_and_preprocess_data(jsonl_path)

    def _load_and_preprocess_data(self, jsonl_path):
        # Load and preprocess data
        texts, labels = [], []
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                    labels.append(int(data['class']))
                except Exception as e:
                    print(f"Failed to process line: {e}")
                    continue

        # Tokenization and attention masks
        tokenized_texts = [self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=512, 
            truncation=True,
            padding=False,  # Do not pad here; handle dynamically later
            return_tensors='pt'
        ) for text in texts]

        # Extract input_ids and attention masks from tokenized outputs, handle padding later
        input_ids = [t['input_ids'].squeeze(0) for t in tokenized_texts]
        attention_masks = [t['attention_mask'].squeeze(0) for t in tokenized_texts]

        # Sort data by length of input_ids to optimize batch processing
        sorted_data = sorted(zip(texts, input_ids, attention_masks, labels), key=lambda x: len(x[1]), reverse=True)

        # Unzip sorted tuples back into separate lists
        texts, input_ids, attention_masks, labels = zip(*sorted_data)

        return (list(texts), list(input_ids), list(attention_masks), list(labels))

    def __len__(self):
        return len(self.data[0])  # All four lists are of the same length

    def __getitem__(self, idx):
        text = self.data[0][idx]
        input_ids = self.data[1][idx]
        attention_mask = self.data[2][idx]
        label = self.data[3][idx]

        # Package and return the data in a dictionary
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
class JSONLDataLoaderFactory(DatasetManagementService, IDatasetOperations, IDataLoaderFactory):
    def __init__(self, db_path='./database/db.sqlite'):
        super().__init__(db_path)

    def search_dataset(self, mode, dataset_name, random_seed=None):
        mode_to_split_ratio = {'train': '80', 'test': '20', "inference": "100"}
        mode_to_total_splits = {'train': 2, 'test': 2, "inference": 1}
        if mode not in mode_to_split_ratio:
            raise ValueError(f"Invalid mode specified: {mode}")

        ratio = mode_to_split_ratio[mode]
        total_splits = mode_to_total_splits[mode]
        dataset_records = self.find_records(dataset_name, total_splits, random_seed)  # Assuming total_splits is always 2
        valid_datasets = [record for record in dataset_records if record[4] == ratio]
        if not valid_datasets:
            raise ValueError(f"No valid datasets found for {dataset_name} in mode {mode} with split ratio {ratio}")
        
        return valid_datasets

    def build_dataset_from_path(self, dataset_path, tokenizer_name):
        tokenizer_path = config_service.get_tokenizer_path(tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return JSONLDataset(dataset_path, tokenizer)

    def make_data_loader(self, mode, dataset_name, random_seed, tokenizer_name, batch_size, shuffle=False, num_workers=0):
        # Search for dataset records using directly provided parameters
        dataset_records = self.search_dataset(mode, dataset_name, random_seed)

        if not dataset_records:
            raise ValueError("No datasets available for the specified configuration.")

        # Get dataset path from the first record found
        dataset_path = dataset_records[0][5]  # Assuming this is the correct index for the dataset path

        # Build dataset from the retrieved path
        dataset = self.build_dataset_from_path(dataset_path, tokenizer_name)

        # Create and return the DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dynamic_padding_collate_fn
        )


    @staticmethod
    def dynamic_padding_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_masks, 'labels': labels}