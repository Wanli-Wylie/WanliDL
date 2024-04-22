import torch
from transformers import BertModel, BertConfig

class BertForBinaryClassification(BertModel):
    def __init__(self, model_path):
        # Load the configuration from the given model path
        config = BertConfig.from_pretrained(model_path)
        
        # Initialize the parent BERT class with the loaded configuration
        super().__init__(config)
        
        # Modify the classifier to be a single output for regression
        # Typically, BERT outputs a vector of size 768, which we map to 1 output for regression
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the parent class's forward to get the sequence output
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask
        )

        # Apply the regression head to the pooled output
        # outputs.pooler_output contains the pooled representation of the entire sequence (CLS token output)
        logits = self.sigmoid(self.classifier(outputs.pooler_output)).squeeze()

        return logits  # Simply return logits in evaluation/inference

