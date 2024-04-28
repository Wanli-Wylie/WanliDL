import torch
from torch import nn
from region_relationship_reasoning import RegionRelationshipReasoning
from global_semantic_reasoning import GlobalSemanticReasoning
from vision_text_generation import VisionTextGeneration
from vision_text_matching import VisionTextMatching

class VSRN(nn.Module):
    def __init__(self, D, bert):
        self.D = D
        super(VSRN, self).__init__()
        self.global_semantic_reasoning = GlobalSemanticReasoning(D)
        self.region_relationship_reasoning = RegionRelationshipReasoning(D)
        self.vision_text_matching = VisionTextMatching(D, bert)
        self.vision_text_generation = VisionTextGeneration(D, bert)
    
    # The forward pass computes the vision feature
    def forward(self, x):
        x = self.region_relationship_reasoning(x)
        x = self.global_semantic_reasoning(x)
        return x
    
    def compute_matching_loss(self, vision_feature, input_ids, attention_mask):
        return self.vision_text_matching.matching_loss(self.vision_text_matching(vision_feature, input_ids, attention_mask))
    
    def compute_generation_loss(self, vision_feature, input_ids, attention_mask):
        return self.vision_text_generation.generate(vision_feature, input_ids, attention_mask)