from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

class Alignment:

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

    class VisionTextGeneration(nn.Module):
        def __init__(self, llama, tokenizer, input_dim):
            super(Alignment.VisionTextGeneration, self).__init__()
            # Project the vision feature to 4096 dimensions
            self.llama = llama
            self.tokenizer = tokenizer
            self.projection = nn.Linear(input_dim, 4096)

        def forward(self, vision_feature):
            # Project vision feature to 768 dimensions

            llama_feature = self.llama.model(inputs_embeds=self.projection(vision_feature))

            predicted_tokens_logits = self.llama.lm_head(llama_feature.last_hidden_state)[:, -1, :].unsqueeze(1)

            # Already logit
            return predicted_tokens_logits

    class SequenceTextGeneration(nn.Module):
        def __init__(self, llama, tokenizer):
            super(Alignment.SequenceTextGeneration, self).__init__()
            self.llama = llama
            self.tokenizer = tokenizer

        def forward(self, previous_tokens):

            lm_result = self.llama(previous_tokens).logits

            predicted_tokens = torch.argmax(lm_result, dim=-1)

            return predicted_tokens

    def __init__(self, llama, input_dim, tokenizer):

        self.llama = llama

        self.tokenizer = tokenizer

        # Freeze all original parameters of the LLaMA model

        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        self.vision_text_generation = self.VisionTextGeneration(self.llama, self.tokenizer, input_dim)

        self.sequence_text_generaton = self.SequenceTextGeneration(self.llama, tokenizer)

        self.vision_text_similarity = self.VisionTextSimilarity(input_dim)

    def autoregression(self, previous_tokens, predicted_tokens_logits):

        predicted_tokens = torch.argmax(predicted_tokens_logits, dim=-1)

        return torch.concat([previous_tokens, predicted_tokens], dim=1)

    def generation_loss(self, vision_feature, reference_text):
        # Generate text from vision feature
        generated_text = self.vision_text_generation(vision_feature)

        loss_list = []

        batch_size, sequence_length = reference_text.shape

        for i in range(sequence_length):

            predicted_tokens_logits = self.sequence_text_generaton(generated_text)

            loss_list.append(F.cross_entropy(predicted_tokens_logits, reference_text[:, i]))
            
            generated_text = self.autoregression(generated_text, predicted_tokens_logits)

        return loss_list

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

if __name__ == "__main__":

    llama = AutoModelForCausalLM.from_pretrained("./pretrained/Meta-Llama-3-8B-Instruct")

    llama_tokenizer = AutoTokenizer.from_pretrained("./pretrained/Meta-Llama-3-8B-Instruct")

    test_string = llama_tokenizer("This is a test string", add_special_tokens=True, return_tensors='pt', truncation=True)

    alignment = Alignment(llama, 2048, llama_tokenizer)

    vision_feature = torch.randn(1, 10, 2048)

    print(alignment.generation_loss(vision_feature, test_string["input_ids"]))

    # vision_feature = torch.randn(2, 2048)
    # text_features = torch.randn(2, 5, 768)

    # loss = alignment.matching_loss(vision_feature, text_features)

    # llama = AutoModel.from_pretrained("./models/llama")

    # llama_text_generation = Alignment.LlamaTextGeneration(llama, input_dim=2048)

    # for name, module in llama_text_generation.named_modules():
    #     try:
    #         print(name, module.weight.shape, module.weight.requires_grad)
    #     except:
    #         print(name, "No weight attribute")
    #         continue
