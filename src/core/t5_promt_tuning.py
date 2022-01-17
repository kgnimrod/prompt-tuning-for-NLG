from os.path import join

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

        model = super().from_pretrained(model_name_or_path, **kwargs)

        #  freeze the transformers model
        for param in model.parameters():
            param.requires_grad = False

        # if a saved soft prompt is loaded, use its embeddings
        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)

        # else create a new soft prompt
        elif number_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                number_tokens=number_tokens, initialize_from_vocab=initialize_from_vocab, random_range=random_range)
        return model

    def set_soft_prompt_embeds(self, soft_prompt_path: str):
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.number_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt. (number_tokens: {self.number_tokens})")

    def initialize_soft_prompt(self, number_tokens: int = 20, initialize_from_vocab: bool = True,
                               random_range: float = 0.5):
        self.number_tokens = number_tokens
        if initialize_from_vocab:
            init_prompt_value = self.shared.weight[:number_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(number_tokens, self.config.d_model).uniform_(-random_range,
                                                                                               random_range)
        self.soft_prompt = torch.nn.Embedding(number_tokens, self.config.d_model)

        # Initialize weight
        self.soft_prompt.weight = torch.nn.parameter.Parameter(init_prompt_value)
        # print(self.soft_prompt.weight.shape)

    # this method appends the learned prompt embeddings to the input ids of the input before the
    # the forward pass is calculated
    def append_learned_embedding_to_input(self, input_ids):
        inputs_embeds = self.shared(input_ids)
        # print(inputs_embeds.shape)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # the shape of the tensor that will be returned will be:
        # [batch_size, max_sequence_length, number_embeddings] -> [8, 600, 512]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        print('shape learned embeds: ' + str(learned_embeds.shape))

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        # print('shape inputs embeds: ' + str(inputs_embeds.shape))
        return inputs_embeds

    # to make sure that padding token ids of the labels are not taken into account by the loss function
    # this method extends the labels tensor by elements that are ignored by the CrossEntropyLoss function
    # this can be done using the ignore_index value -100
    def extend_labels(self, labels, ignore_index=-100):
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        number_of_batches = labels.shape[0]

        # print('number batches: ' + str(n_batches))

        # return a new tensor of shape [number_of_batches, number_tokens+labels]
        # that is filled with the ignore_index value (-100)
        return torch.cat([torch.full((number_of_batches, self.number_tokens), ignore_index).to(self.device), labels],
                         dim=1)

    def extend_attention_mask(self, attention_mask):
        # prepend a new dimension (1) to the shape of attention_mask in case it is one dimensional
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # get the number of batches
        number_of_batches = attention_mask.shape[0]

        # return a new tensor of shape [number_of_batches, number_tokens+attention_mask] that is filled with the ones
        return torch.cat([torch.full((number_of_batches, self.number_tokens), 1).to(self.device), attention_mask],
                         dim=1)

    def save_soft_prompt(self, filename: str = "soft_prompt.model"):
        torch.save(self.soft_prompt, join('t5-tuning', 'soft_prompts', filename))

    # def forward(
    #         self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None,
    #         position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
    #         encoder_attention_mask=None, labels=None, use_cache=None, output_attentions=None,
    #         output_hidden_states=None, return_dict=None):

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.append_learned_embedding_to_input(input_ids).to(self.device)
            else:
                inputs_embeds = self.append_learned_embedding_to_input(decoder_input_ids).to(self.device)

        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = inputs_embeds

        if labels is not None:
            labels = self.extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self.extend_attention_mask(attention_mask).to(self.device)

        print("T5 Prompt Tuning forward: input embeds: ", inputs_embeds)
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
