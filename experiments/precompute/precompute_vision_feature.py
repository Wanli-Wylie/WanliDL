import json
from typing import Dict, Optional
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from PIL import Image
import os
from pathlib import Path
import glob
from tqdm import tqdm

import sys
sys.path.append('.')
from WanliDL.services.feature_store import FasterRCNNFeatureStore
db_path = './datasets/vision/COCO/images/val2014/features.sqlite'
db = FasterRCNNFeatureStore(db_path)

def process_images(image_dir, output_dir, model):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    transform = T.Compose([T.ToTensor()])

    image_paths = glob.glob(os.path.join(image_dir, '*'))
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Check if the record already exists
        if db.record_exists(image_path):
            continue  # Skip the file if a record already exists

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to('cuda')

        with torch.no_grad():
            predictions = model([image_tensor])

        num_objects_detected = len(predictions[0]['boxes'])
        image_shape = f"{image.width}x{image.height}"
        output_dict = {
            'boxes': predictions[0]['boxes'].tolist(),
            'labels': predictions[0]['labels'].tolist(),
            'scores': predictions[0]['scores'].tolist(),
        }

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        feature_path = os.path.join(output_dir, f'{base_filename}.json')
        with open(feature_path, 'w') as f:
            json.dump(output_dict, f)

        db.add_feature_record(image_shape, image_path, num_objects_detected, feature_path)

if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.eval()
    model.cuda()
    image_dir = './datasets/vision/COCO/images/val2014/val2014'
    output_dir = './datasets/vision/COCO/images/val2014/features'
    process_images(image_dir, output_dir, model)