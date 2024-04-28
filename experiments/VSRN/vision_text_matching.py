import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTextMatching(nn.Module):
    def __init__(self, input_dim, bert):
        super(VisionTextMatching, self).__init__()
        # Initialize the projection layer
        self.proj = nn.Linear(768, input_dim)
        self.bert = bert

    def forward(self, vision_feature, input_ids, attention_mask):

        with torch.no_grad():
            text_feature = self.bert(input_ids, attention_mask=attention_mask).pooler_output

        # Project text feature to the same dimension as vision feature
        text_feature = self.proj(text_feature)  # Output shape: B x projection_dim

        # Ensure both features are in the same shape B x projection_dim
        assert vision_feature.size(1) == text_feature.size(1), "Feature dimensions do not match"

        # Compute similarity by dot product or cosine similarity
        similarity = F.cosine_similarity(vision_feature.unsqueeze(1), text_feature.unsqueeze(0), dim=2)  # B x B
        return similarity

    @classmethod
    def matching_loss(cls, similarity, margin=0.2):
        # Convert similarity to distance (assuming higher similarity means lower distance)
        distance = -similarity

        # Calculate the maximum distance for matched pairs
        positive_distances = distance.diagonal()  # Distance of each element to itself

        # Create a mask to ignore the diagonal elements when calculating negative distances
        eye = torch.eye(distance.size(0), device=distance.device)
        max_negative_distances = distance.masked_fill(eye.bool(), float('-inf')).max(dim=1).values

        # Calculate triplet hinge loss
        loss = F.relu(positive_distances - max_negative_distances + margin).sum()
        return loss
    