import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class VisionTextGeneration(nn.Module):
    def __init__(self, feature_dim, bert_model, hidden_dim=None, vocab_size=None):
        super(VisionTextGeneration, self).__init__()
        if hidden_dim is None:
            hidden_dim = bert_model.config.hidden_size
        if vocab_size is None:
            vocab_size = bert_model.config.vocab_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # BERT model for embedding
        self.bert_model = bert_model
        
        # GRU layer
        self.gru = nn.GRU(self.bert_model.config.hidden_size, hidden_dim, batch_first=True)
        
        # Linear layer to map hidden state output to vocabulary size
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        # Initial features to hidden state
        self.features_to_hidden = nn.Linear(feature_dim, hidden_dim)

    def forward(self, input_ids, attention_mask, features, states=None):
        # Generate embeddings from BERT
        with torch.no_grad():  # Optionally freeze BERT during training
            embeddings = self.bert_model(input_ids, attention_mask=attention_mask).pooler_output.unsqueeze(1)  # Only take the last hidden states
        
        # Initialize hidden state with image features if states is None
        hidden = self.features_to_hidden(features).unsqueeze(0) if states is None else states
        
        # GRU processing
        gru_out, hidden = self.gru(embeddings, hidden)
        output = self.linear(gru_out)
        
        return output, hidden

    def generate(self, vision_feature, input_ids, attention_mask):
        results = []
        hidden = None
        
        for i in range(input_ids.size(1)):
            # In the first iteration, use the vision feature to initialize the hidden state
            output, hidden = self.forward(input_ids[:, :i+1], attention_mask[:, :i+1], vision_feature, hidden)

            output = F.softmax(output[:, -1, :], dim=-1)
            _, top_idx = torch.max(output, dim=-1)

            loss = F.cross_entropy(output, input_ids[:, i])

            results.append(loss)

        return results

if __name__ == "__main__":
    bert_tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-uncased')
    bert_model = BertModel.from_pretrained('./pretrained/bert-base-uncased')

    model = VisionTextGeneration(feature_dim=2048, bert_model=bert_model)

    # Test the model
    features = torch.rand(2, 2048)
    input_texts = ["Hello world my cat", "Look at this"]
    input_ids = bert_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids

