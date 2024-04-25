import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class RegionRelationshipReasoning(nn.Module):
    def __init__(self, D):
        super(RegionRelationshipReasoning, self).__init__()
        # Number of regions and dimension of embeddings
        
        self.proj = nn.Linear(4096, D, bias=False)

        self.D = D

        # Embedding layers for each image region vi and vj
        self.phi = nn.Linear(D, D, bias=False)
        self.pho = nn.Linear(D, D, bias=False)
        
        # GCN weights
        self.Wg = nn.Linear(D, D, bias=False)
        self.Wr = nn.Linear(D, D, bias=True)
    
    def forward(self, V):
        # V: Input features of regions (batch_size, num_regions, D)
        
        V = self.proj(V)

        # Calculate embeddings
        Phi_V = self.phi(V)  # Shape: (batch_size, num_regions, D)
        Pho_V = self.pho(V)  # Shape: (batch_size, num_regions, D)
        
        # Calculate the affinity matrix R
        R = torch.bmm(Phi_V, Pho_V.transpose(1, 2))  # Transpose for matrix multiplication
        
        # Normalize R row-wise
        R = F.softmax(R, dim=2)
        
        # Apply GCN
        RVWg = torch.bmm(R, self.Wg(V))
        
        # Add residual connection
        V_star = self.Wr(RVWg) + V  # Residual connection
        
        return V_star
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    with open("./datasets/vision/Flickr30k/vision_features/2753531542.json") as f:
        test_feature = json.load(f)

    test_feature['region_features'] = torch.tensor(test_feature['region_features']).unsqueeze(0)

    print(test_feature['region_features'].shape)
    
    # Create a model
    model = RegionRelationshipReasoning(D=2048)
    
    print(model(test_feature['region_features']).shape)
    