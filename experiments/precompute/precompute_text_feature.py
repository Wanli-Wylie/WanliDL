import json
from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('.')
from WanliDL.services.feature_store import BertFeatureStore
# Assuming BertFeatureStore has been imported from another module

class BertCaptionProcessor:
    def __init__(self, model_path='./pretrained/bert-base-uncased', db_path='./datasets/vision/Flickr30k/features.sqlite'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()  # set model to evaluation mode
        if torch.cuda.is_available():
            self.model.cuda()
        self.db = BertFeatureStore(db_path)

    def preprocess_captions(self, dataset_file, output_dir):
        """
        Process all captions in the dataset, extracting BERT embeddings and saving them,
        along with recording the information in the feature database.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(dataset_file, 'r', encoding='utf-8') as file:
            data = file.readlines()

        # Process each line
        for line in tqdm(data, desc="Processing captions"):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            image_file, caption = parts[0].split('#')[0], parts[1]

            image_id = Path(image_file).stem
            
            # Create output file path
            output_file = Path(output_dir) / f'{image_id}.jsonl'
            output_file_path = str(output_file)

            # Skip processing if the feature record already exists
            if self.db.record_exists(output_file_path):
                continue

            # Tokenize and encode the caption
            inputs = self.tokenizer(caption, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get BERT outputs
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract the <CLS> token's embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

            # Append to JSONL file
            with open(output_file, 'a') as f:
                json.dump(cls_embedding, f)
                f.write('\n')

            # Add a record in the database
            self.db.add_feature_record(image_id, output_file_path)

if __name__ == '__main__':
    dataset_file = 'datasets/vision/Flickr30k/results_20130124.token'  # Path to your dataset file
    output_dir = 'datasets/vision/Flickr30k/text_features'  # Directory to store output JSONL files
    processor = BertCaptionProcessor()
    processor.preprocess_captions(dataset_file, output_dir)
