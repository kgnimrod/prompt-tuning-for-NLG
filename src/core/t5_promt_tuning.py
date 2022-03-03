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

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, soft_prompt_path: str = None, number_tokens: int = None,
                        initialize_from_vocab: bool = True, random_range: float = 0.5, **kwargs):

        print("model created with T5PromptTuning.from_pretrained")
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
            init_prompt_value = (
                self.get_input_embeddings().weight[:number_tokens].clone().detach()
            )
        else:
            init_prompt_value = torch.FloatTensor(number_tokens, self.config.d_model)\
                .uniform_(-random_range, random_range)

        self.soft_prompt = torch.nn.Embedding(number_tokens, self.config.d_model)
        # Initialize weight
        self.soft_prompt.weight = torch.nn.parameter.Parameter(init_prompt_value)

    def get_soft_params(self):
        return self.soft_prompt

    # this method appends the learned prompt embeddings to the input ids of the input before forward pass is calculated
    def extend_inputs(self, input_ids):
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

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

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            to_encoder_only=False,
    ):
        if input_ids is not None:
            inputs_embeds = self.extend_inputs(input_ids).to(self.device)

        if labels is not None:
            labels = self.extend_labels(labels).to(self.device)

        # for training, extend the attention mask to include input embeddings, but not for inference,
        # where greedy search only requires encoder outputs and decoder_input ids and the shape needs to match
        if attention_mask is not None:
            attention_mask = self.extend_attention_mask(attention_mask).to(self.device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = self.extend_attention_mask(
                    decoder_attention_mask
                ).to(self.device)

        if to_encoder_only:
            return self.encoder(inputs_embeds=inputs_embeds, return_dict=True)

        # for inference (i.e. generate) - build pipeline for generate function
        if decoder_input_ids is not None:
            return super().forward(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                use_cache=use_cache,
                return_dict=return_dict,
            )

        # for training
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class T5PromptTuningLM(T5PromptTuning, T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
