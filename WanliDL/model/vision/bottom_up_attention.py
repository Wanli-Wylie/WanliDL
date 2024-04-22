import torch
from torch import nn
import PIL
import numpy as np
from torchvision.ops import roi_align, nms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

class BottomUpAttention(nn.Module):
    def __init__(self, faster_rcnn, D=2048, iou_threshold=0.7, confidence_threshold=0.3, top_rois=36, use_fc=False):

        super(BottomUpAttention, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.D = D
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.top_rois = top_rois
        self.use_fc = use_fc
        if self.use_fc:
            self.fc = torch.nn.Linear(256, D)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        # Perform a forward pass to get detections and backbone features
        original_image_sizes = [img.shape[-2:] for img in images]
        outputs = self.faster_rcnn(images)
        features = self.faster_rcnn.backbone(images)  # Extract feature maps
        
        batched_features = []
        for i, output in enumerate(outputs):
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
            for level in range(4):  # Assuming FPN outputs P2 to P5 maps
                level_boxes = boxes[levels == level]
                if len(level_boxes) == 0:
                    continue
                level_features = features[str(level)][i:i+1]  # Selecting the feature map for current level
                
                rois = torch.cat((torch.zeros(len(level_boxes), 1, device=level_boxes.device), level_boxes), dim=1)
                region_features = roi_align(level_features, rois, output_size=(7, 7))
                
                # Apply average pooling
                pooled_features = self.avg_pool(region_features).view(region_features.size(0), -1)
                
                # Optionally apply the fully connected layer
                if self.use_fc:
                    pooled_features = self.fc(pooled_features)
                
                batched_features.append(pooled_features)

        return torch.cat(batched_features) if batched_features else torch.tensor([])

if __name__ == "__main__":

    image = torch.from_numpy(np.array(PIL.Image.open('test.jpg'))).permute(2, 0, 1).unsqueeze(0).float()

    fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    fasterrcnn.eval()

    bottom_up_attention = BottomUpAttention(fasterrcnn)

    bottom_up_attention.eval()

    output = bottom_up_attention(image)

    print(output.shape)