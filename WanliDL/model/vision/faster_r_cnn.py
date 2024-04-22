from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import torch


class FasterRCNN:
    def __init__(self, backbone, num_classes, anchor_sizes, pretrained):
        self.backbone = backbone
        self.num_classes = num_classes
        self.anchor_sizes = anchor_sizes
        self.pretrained = pretrained

    def forward(self, images):
        # Forward pass through the backbone
        features = self.backbone(images)
        
        # Perform the rest of the Faster R-CNN operations

if __name__ == "__main__":

    random_input = torch.rand(3, 3, 224, 224)

    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    model.eval()
    with torch.no_grad():
        output = model(random_input)
        print(output)