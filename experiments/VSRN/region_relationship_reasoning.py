import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionRelationshipReasoning(nn.Module):
    def __init__(self, D):
        super(RegionRelationshipReasoning, self).__init__()
        # Number of regions and dimension of embeddings
        
        self.D = D

        # Embedding layers for each image region vi and vj
        self.phi = nn.Linear(D, D, bias=False)
        self.pho = nn.Linear(D, D, bias=False)
        
        # GCN weights
        self.Wg = nn.Linear(D, D, bias=False)
        self.Wr = nn.Linear(D, D, bias=True)
    
    def forward(self, V):
        # V: Input features of regions (batch_size, num_regions, D)
        
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
    
    # Create a model
    model = RegionRelationshipReasoning(D=128)
    
    # Generate some random input data
    V = torch.randn(2, 10, 128)  # (batch_size=2, num_regions=10, embedding_dim=128
    
    # Forward pass
    V_star = model(V)
    
    print("Input shape:", V.shape)
    print("Output shape:", V_star.shape)