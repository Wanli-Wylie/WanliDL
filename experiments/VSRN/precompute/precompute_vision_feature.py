# Import necessary libraries for image processing
import json
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.ops import roi_align, nms
from PIL import Image
import os
from pathlib import Path
import glob
from tqdm import tqdm
import math

import sys
sys.path.append('.')
# Import database management functionalities
from WanliDL.services.feature_store import FasterRCNNFeatureStore

# Parameters for the region feature filtering
iou_threshold = 0.7
confidence_threshold = 0.3
top_rois = 36

def process(boxes, scores, image_features):
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep][:top_rois]
    _, _, height, width = image_features['0'].shape
    box_sizes = ((boxes[:, 2:] - boxes[:, :2]) ** 2).sum(dim=1).sqrt()
    levels = torch.floor(4 + torch.log2(box_sizes / math.sqrt(height * width))).int()
    levels = torch.clamp(levels, min=2, max=5) - 2

    batched_features = []
    for level in set(levels.cpu().tolist()):
        level_boxes = boxes[levels == level]
        rois = torch.cat((torch.zeros(len(level_boxes), 1, device=level_boxes.device), level_boxes / 4 / 2 ** level), dim=1)
        level_features = image_features[str(level)]
        region_features = roi_align(level_features, rois, output_size=4)
        batched_features.append(region_features)

    batched_features = torch.cat(batched_features, dim=0).flatten(1)
    return batched_features.cpu().numpy()

def process_images(image_dir, output_dir, model, feature_store):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    transform = T.Compose([T.ToTensor()])
    image_paths = glob.glob(os.path.join(image_dir, '*'))
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).to('cuda')

            with torch.no_grad():
                predictions = model([image_tensor])
                image_features = model.backbone(image_tensor.unsqueeze(0))

            scores = predictions[0]['scores']
            boxes = predictions[0]['boxes']

            keep = scores > confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]

            region_features = process(boxes, scores, image_features)
            num_of_regions = region_features.shape[0]

            output_dict = {
                'boxes': boxes.tolist(),
                'scores': scores.tolist(),
                'region_features': region_features.tolist(),
                'num_of_regions': num_of_regions,
                'parameters': {
                    'iou_threshold': iou_threshold,
                    'confidence_threshold': confidence_threshold,
                    'top_rois': top_rois
                }
            }

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            feature_path = os.path.join(output_dir, f'{base_filename}.json')
            
            with open(feature_path, 'w') as f:
                json.dump(output_dict, f)

            # Add feature record to the database
            if not feature_store.record_exists(image_path):
                feature_store.add_feature_record(str(image_tensor.shape), image_path, num_of_regions, feature_path)
        except Exception as e:
            print(f"Failed to process image: {e}")
            continue

if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True).eval().cuda()
    image_dir = './datasets/vision/MSCOCO/train2014'
    output_dir = './datasets/vision/MSCOCO/train2014/vision_features'
    
    feature_store = FasterRCNNFeatureStore(db_path='./datasets/vision/MSCOCO/train2014_features.sqlite')
    try:
        process_images(image_dir, output_dir, model, feature_store)
    finally:
        feature_store.close_connection()