from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class Alignment:

    class LlamaTextGeneration(nn.Module):
        def __init__(self, llama, input_dim):
            super(Alignment.LlamaTextGeneration, self).__init__()
            self.llama = llama

            # Assume the embedding size of the LLaMA model is known (4096 in your case)
            self.adapter_layer = nn.Linear(input_dim, 4096)

            # The name of the adapter layer should be the same as the name of the embedding layer in the LLaMA model
            self.llama.embed_tokens = self.adapter_layer

            # Freeze all original parameters of the LLaMA model
            for name, param in self.llama.named_parameters():
                param.requires_grad = False

            self.llama.embed_tokens.requires_grad = True

        def forward(self, x):
            # Pass input through adapter layer then through the rest of the LLaMA model
            return self.llama(x)

        def get_trainable_parameters(self):
            # Returns an iterator over the adapter layer parameters, which are trainable
            return self.adapter_layer.parameters()

    class VisionTextSimilarity(nn.Module):
        def __init__(self, input_dim):
            super(Alignment.VisionTextSimilarity, self).__init__()
            # Project the vision feature to 768 dimensions
            self.projection = nn.Linear(input_dim, 768)

        def forward(self, vision_feature, text_feature):
            # vision_feature shape: B, input_dim
            # text_feature shape: B, 5, 768

            # Project vision feature to 768 dimensions
            vision_projected = self.projection(vision_feature)  # Output shape: B, 768

            # We need to calculate the pairwise similarity measure between each pair of vision and text features
            # vision_projected is B x 768, text_feature is B x 5 x 768
            # To perform batch matrix multiplication we adjust shapes to:
            # vision_projected: B x 1 x 768, text_feature: B x 768 x 5
            vision_projected = vision_projected.unsqueeze(1)  # B x 1 x 768
            text_feature = text_feature.permute(0, 2, 1)  # B x 768 x 5

            # Batch matrix multiplication to get similarity scores, result is B x 1 x 5
            similarity = torch.bmm(vision_projected, text_feature)  # Output shape: B x 1 x 5

            # To get pairwise similarity measures B x B x 5, we compute similarity for every combination
            # We expand similarity to compare with every other, resulting in B x B x 5
            pairwise_similarity = similarity.expand(-1, vision_feature.shape[0], -1)  # Output shape: B x B x 5

            return pairwise_similarity

    def __init__(self, llama, input_dim, tokenizer_path):
#        self.llama_text_generaton = self.LlamaTextGeneration(llama, input_dim)
        self.vision_text_similarity = self.VisionTextSimilarity(input_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _compute_matching_loss(self, pairwise_similarity, margin=1.0):
        # Convert similarity to distance (assuming higher similarity means lower distance)
        distance = -pairwise_similarity

        # Get the max distance for matched pairs (d(A, P))
        positive_distances = distance.diagonal(dim1=0, dim2=1).max(dim=0).values  # Max across the 5 captions

        # Create a mask to ignore the diagonal elements when calculating negative distances
        eye = torch.eye(distance.size(0), device=distance.device)
        max_negative_distances = distance.masked_fill(eye.bool().unsqueeze(2), float('-inf')).max(dim=1).values.max(dim=1).values

        # Calculate triplet hinge loss
        loss = F.relu(positive_distances - max_negative_distances + margin).mean()

        return loss

    def matching_loss(self, vision_feature, text_features, margin=1.0):
        # Compute pairwise similarity between vision and text features
        pairwise_similarity = self.vision_text_similarity(vision_feature, text_features)

        # Compute matching loss
        loss = self._compute_matching_loss(pairwise_similarity, margin)

        return loss
    
    # def generation_loss(self, vision_feature, ground_truth_text):
    #     # Generate text features from vision feature
    #     generated_text = self.llama_text_generaton(vision_feature)

    #     # Tokenize the generated text
    #     generated_text = self.tokenizer(ground_truth_text, return_tensors='pt', padding=True, truncation=True)

    #     # Compute the loss between generated text and ground truth text features
    #     loss = F.cross_entropy(generated_text, text_features)

    #     return loss

if __name__ == "__main__":
    alignment = Alignment(None, 2048, "./pretrained/bert-base-uncased")

    vision_feature = torch.randn(2, 2048)
    text_features = torch.randn(2, 5, 768)

    loss = alignment.matching_loss(vision_feature, text_features)

    # llama = AutoModel.from_pretrained("./models/llama")

    # llama_text_generation = Alignment.LlamaTextGeneration(llama, input_dim=2048)

    # for name, module in llama_text_generation.named_modules():
    #     try:
    #         print(name, module.weight.shape, module.weight.requires_grad)
    #     except:
    #         print(name, "No weight attribute")
    #         continue
