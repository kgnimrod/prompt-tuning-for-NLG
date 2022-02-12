from transformers import T5ForConditionalGeneration
import torch


class T5PromptTuning(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.number_tokens = None
        self.soft_prompt = None
        self.config = config
        self.model_dim = config.d_model
        self.shared = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.device = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, soft_prompt_path: str = None, number_tokens: int = None,
                        initialize_from_vocab: bool = True, random_range: float = 0.5, **kwargs):

        model = super().from_pretrained(model_name_or_path, **kwargs)

        #  freeze the transformers model
        for param in model.parameters():
            param.requires_grad = False

        # if a saved soft prompt is loaded, use its embeddings
        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path=soft_prompt_path)
        # else create a new soft prompt
        elif number_tokens is not None:
            model.initialize_soft_prompt(
                number_tokens=number_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range
            )
        return model

    def set_soft_prompt_embeds(self, soft_prompt_path):
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.number_tokens = self.soft_prompt.shape[0]

    def initialize_soft_prompt(self, number_tokens: int = 20, initialize_from_vocab: bool = True, random_range: float = 0.5):
        self.number_tokens = number_tokens
        if initialize_from_vocab:
            init_prompt_value = self.shared.weight[:number_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(number_tokens, self.config.d_model)\
                .uniform_(-random_range, random_range)

        print(init_prompt_value.shape)
        print(self.shared.weight.shape)

        # Initialize weight
        self.soft_prompt = torch.nn.parameter.Parameter(init_prompt_value)

    def get_soft_params(self):
        return self.soft_prompt

    # this method appends the learned prompt embeddings to the input ids of the input before forward pass is calculated
    def embed_tokens(self, input_ids):
        inputs_embeds = self.shared(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # the shape of the tensor that will be returned will be:
        # [batch_size, max_sequence_length, number_embeddings] -> [8, 600, 512]
        learned_embeds = self.soft_prompt.repeat(inputs_embeds.size(0), 1, 1)
        return torch.cat([learned_embeds, inputs_embeds], dim=1)

    # to make sure that padding token ids of the labels are not taken into account by the loss function
    # this method extends the label's tensor by elements that are ignored by the CrossEntropyLoss function
    # this can be done using the ignore_index value -100
    def extend_labels(self, labels, ignore_index=-100):
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        number_of_batches = labels.shape[0]

        # return a new tensor of shape [number_of_batches, number_tokens+labels]
        # that is filled with the ignore_index value (-100)
        return torch.cat(
            [torch.full((number_of_batches, self.number_tokens), ignore_index).to(self.device), labels],
            dim=1
        )

    def extend_attention_mask(self, attention_mask):
        # prepend a new dimension (1) to the shape of attention_mask in case it is one dimensional
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # get the number of batches
        number_of_batches = attention_mask.shape[0]

        # return a new tensor of shape [number_of_batches, number_tokens+attention_mask] that is filled with the ones
        return torch.cat(
            [torch.full((number_of_batches, self.number_tokens), 1).to(self.device), attention_mask],
            dim=1
        )

