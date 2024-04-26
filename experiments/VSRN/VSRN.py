import torch
from torch import nn
from region_relationship_reasoning import RegionRelationshipReasoning
from global_semantic_reasoning import GlobalSemanticReasoning
from vision_text_generation import VisionTextGeneration
from vision_text_matching import VisionTextMatching

class VSRN(nn.Module):
    def __init__(self, D, bert):
        self.D = D
        self.global_semantic_reasoning = GlobalSemanticReasoning(D)
        self.region_relationship_reasoning = RegionRelationshipReasoning(D)
        self.vision_text_matching = VisionTextMatching(D)
        self.vision_text_generation = VisionTextGeneration(bert, D)
    
    # The forward pass computes the vision feature
    def forward(self, x):
        x = self.region_relationship_reasoning(x)
        x = self.global_semantic_reasoning(x)
        return x
    
    def compute_matching_loss(self, vision_feature, text_feature):
        return self.vision_text_matching(vision_feature, text_feature)
    
    def compute_generation_loss(self, vision_feature, input_ids, attention_mask):
        return self.vision_text_generation(vision_feature, input_ids, attention_mask)