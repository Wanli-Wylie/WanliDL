import torch
import torch.nn as nn

class GlobalSemanticReasoning(nn.Module):
    def __init__(self, D):
        super(GlobalSemanticReasoning, self).__init__()
        # Initialize the GRU
        # input_dim is D
        self.gru = nn.GRU(D, D, batch_first=True)

    def forward(self, x):
        # x should have shape (B, k, D)
        # We only need the hidden state from the last timestep
        _, h_n = self.gru(x)  # h_n shape is (1, B, hidden_dim)
        # Squeeze the first dimension (num_layers*num_directions) for the final output
        return h_n.squeeze(0)
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    v = torch.rand(2, 5, 2048)

    model = GlobalSemanticReasoning(D=2048)

    output = model(v)

    print(output.shape)