import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTextMatching(nn.Module):
    def __init__(self, input_dim):
        super(VisionTextMatching, self).__init__()
        # Initialize the projection layer
        self.proj = nn.Linear(input_dim, 768)

    def forward(self, vision_feature, text_feature):
        # Project vision feature to 768 dimensions
        vision_projected = self.proj(vision_feature)  # Output shape: B, 768
        vision_projected = vision_projected.unsqueeze(1)  # B x 1 x 768
        text_feature = text_feature.permute(0, 2, 1)  # B x 768 x 5

        # Expand both vision and text features to compare each against each
        vision_projected_expanded = vision_projected.unsqueeze(2).expand(-1, -1, text_feature.size(0) * text_feature.size(1))  # B x 768 x (B*5)
        text_feature_flattened = text_feature.view(-1, 768)  # (B*5) x 768
        text_feature_expanded = text_feature_flattened.unsqueeze(0).expand(vision_feature.size(0), -1, -1)  # B x (B*5) x 768

        # Compute similarity by dot product or cosine similarity across expanded dimensions
        similarity = torch.bmm(vision_projected_expanded, text_feature_expanded.transpose(1, 2))  # B x B x 5
        return similarity

    @classmethod
    def matching_loss(cls, similarity, margin=1.0):
        # Convert similarity to distance (assuming higher similarity means lower distance)
        distance = -similarity

        # Calculate the maximum distance for matched pairs
        positive_distances = distance.diagonal(dim1=0, dim2=1).max(dim=0).values  # Max across the 5 captions

        # Create a mask to ignore the diagonal elements when calculating negative distances
        eye = torch.eye(distance.size(0), device=distance.device)
        max_negative_distances = distance.masked_fill(eye.bool().unsqueeze(2), float('-inf')).max(dim=1).values.max(dim=1).values

        # Calculate triplet hinge loss
        loss = F.relu(positive_distances - max_negative_distances + margin).mean()
        return loss
