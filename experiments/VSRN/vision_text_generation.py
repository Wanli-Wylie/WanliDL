import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTextGeneration(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, embedding_dim):
        super(VisionTextGeneration, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        
        # Linear layer to map hidden state output to vocabulary size
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        # Embedding layer for words
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initial features to hidden state
        self.features_to_hidden = nn.Linear(feature_dim, hidden_dim)

    def forward(self, input_word, features, states=None):
        # Initialize hidden state with image features if states is None
        hidden = self.features_to_hidden(features).unsqueeze(0) if states is None else states
        
        # Get embeddings for the input word
        word_embed = self.embeddings(input_word).unsqueeze(0)
        gru_out, hidden = self.gru(word_embed, hidden)
        output = self.linear(gru_out.squeeze(0))
        
        return output, hidden

    def generate(self, features, vocab, max_length=20):
        results = []
        input_word = torch.tensor([vocab.start_idx], device=features.device)  # Assuming there is a start token index defined
        hidden = None

        for _ in range(max_length):
            output, hidden = self.forward(input_word, features, hidden)
            output = F.softmax(output, dim=1)
            _, top_idx = torch.max(output, dim=1)
            input_word = top_idx
            results.append(top_idx.item())
            if top_idx == vocab.end_idx:  # Assuming there is an end token index defined
                break

        return results
