import torch
import torch.nn as nn
from torchvision.ops import roi_align, nms

class BottomUpAttention(nn.Module):
    def __init__(self, D=2048, iou_threshold=0.7, confidence_threshold=0.3, top_rois=36, num_workers=8):
        super(BottomUpAttention, self).__init__()
        self.D = D
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.top_rois = top_rois
        self.fc = nn.Linear(2048, D).cuda() 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers

    def forward(self, detections, feature_maps):
        # Container for all batch features after processing
        batched_features = []

        # Process each image in the batch
        for output, image_features in zip(detections, feature_maps):
            scores = output['scores']
            boxes = output['boxes']
            labels = output['labels']

            # Filter proposals based on confidence threshold
            keep = scores > self.confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Apply NMS to reduce box overlap
            keep = nms(boxes, scores, self.iou_threshold)
            boxes = boxes[keep][:self.top_rois]

            # Calculate which FPN level each box should be assigned to
            box_sizes = ((boxes[:, 2:] - boxes[:, :2]) ** 2).sum(dim=1).sqrt()  # Diagonal size of each box
            levels = torch.floor(4 + torch.log2(box_sizes / 224 * 4)).int()  # 224 is the typical scale for level 0, scale changes by 2**level
            levels = torch.clamp(levels, min=2, max=5) - 2  # Adjust levels between 0 and 3 for P2 to P5

            # Process each level's boxes
            for level in range(4):  # Assuming FPN outputs '0' to '3' maps
                level_boxes = boxes[levels == level]
                if len(level_boxes) == 0:
                    continue
                level_features = image_features[str(level)][None]  # Selecting the feature map for current level

                # Create RoIs for roi_align, and move tensors to GPU
                rois = torch.cat((torch.zeros(len(level_boxes), 1, device=level_boxes.device), level_boxes), dim=1)
                region_features = roi_align(level_features, rois, output_size=(7, 7)).to('cuda' if torch.cuda.is_available() else 'cpu')

                # Apply average pooling
                pooled_features = self.avg_pool(region_features).view(region_features.size(0), -1)

                # Optionally apply the fully connected layer
                pooled_features = self.fc(pooled_features)

                batched_features.append(pooled_features)

        # Concatenate all features and send to GPU if available
        if batched_features:
            return torch.cat(batched_features, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.tensor([], device='cuda' if torch.cuda.is_available() else 'cpu')

# Example usage setup
# detections = [{'scores': scores_tensor, 'boxes': boxes_tensor, 'labels': labels_tensor}, ...]
# feature_maps = [{'0': feature_map_0, '1': feature_map_1, '2': feature_map_2, '3': feature_map_3}, ...]
# bottom_up_attention = BottomUpAttention()
# result = bottom_up_attention(detections, feature_maps)
