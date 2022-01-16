import os
import warnings
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from itertools import chain
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from transformers.optimization import Adafactor, AdamW
from IPython.display import HTML, display

def progress(loss,value, max=100):
    return HTML(""" Batch loss :{loss}
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(loss=loss,value=value, max=max))

def trainer(model, num_batches, inputs, labels, optimizer, challenge_name, model_name):
    # Set the model in training mode
    model.train()

    #torch.cuda.empty_cache()

    loss_per_10_steps=[]
    for epoch in range(1,epochs+1):
      print('Running epoch: {}'.format(epoch))
      running_loss=0

      out = display(progress(1, num_batches+1), display_id=True)
      for i in range(num_batches):

        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(input_ids=inputs[i], labels=labels[i])
        loss = outputs.loss
        loss_num=loss.item()
        logits = outputs.logits
        running_loss+=loss_num
        if i%10 == 0: loss_per_10_steps.append(loss_num)
        out.update(progress(loss_num,i, num_batches+1))

        # calculating the gradients
        loss.backward()

        #updating the params
        optimizer.step()

      running_loss=running_loss/int(num_batches)
      print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))

      # plot the loss
      steps = [i for i in range(len(loss_per_10_steps))]
      plt.plot(steps, loss_per_10_steps)
      plt.title(f'Loss curve for the {challenge_name} challenge trained for {epochs} epochs on T5-{model_name}')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.show()
    return model

def optimizer_adafactor(model,
                        lr=1e-3,  # default values for adafactor
                        eps=(1e-30, 1e-3),  # default values for adafactor
                        clip_threshold=1.0,  # default values for adafactor
                        decay_rate=-0.8,  # default values for adafactor
                        beta1=None,  # default values for adafactor
                        weight_decay=0.0,  # default values for adafactor
                        relative_step=False,
                        scale_parameter=False,
                        warmup_init=False):
    return Adafactor(
        [model_t5_small.get_soft_params()],
        lr=lr,
        eps=eps,
        clip_threshold=clip_threshold,
        decay_rate=decay_rate,
        beta1=beta1,
        weight_decay=weight_decay,
        relative_step=relative_step,
        scale_parameter=scale_parameter,
        warmup_init=warmup_init
    )
def create_list_of_batches(batch_size, num_batches, data, tokenizer):
# Create List of batches for inputs and labels
    inputs = []
    labels = []
    for i in range(num_batches):
        input_batch=[]
        label_batch=[]
        for index,row in data[i*batch_size:i*batch_size+batch_size].iterrows():
#          input_batch.append('translate from Graph to Text: '+row['input_text']+'</s>')
#          label_batch.append(row['target_text']+'</s>')

          input_batch.append('translate from Graph to Text: '+row['input_text'])
          label_batch.append(row['target_text'])

        input_batch=tokenizer.batch_encode_plus(input_batch,padding=True, max_length=400, return_tensors='pt', return_attention_mask=True)
        #print('input_batch shape: ' + str(input_batch.shape))
        label_batch=tokenizer.batch_encode_plus(label_batch,padding=True, max_length=400, return_tensors='pt', return_attention_mask=True)
        #print('label_batch shape: ' + str(label_batch.shape))

        input_batch=input_batch.to(dev)
        label_batch=label_batch.to(dev)

        inputs.append(input_batch)
        labels.append(label_batch)
    return inputs, labels

class T5PromptTuning(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

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
            model.initialize_soft_prompt(number_tokens=number_tokens, initialize_from_vocab=initialize_from_vocab,
                                         random_range=random_range)
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
        # self.soft_prompt = torch.nn.Embedding(number_tokens, self.config.d_model)

        print(init_prompt_value.shape)
        print(self.shared.weight.shape)
        # Initialize weight
        self.soft_prompt = torch.nn.parameter.Parameter(init_prompt_value)
        # print(self.soft_prompt.weight.shape)

    def get_soft_params(self):
        return self.soft_prompt

    # this method appends the learned prompt embeddings to the input ids of the input before the forward pass is calculated
    def append_learned_embedding_to_input(self, input_ids):
        inputs_embeds = self.shared(input_ids)

        if len(list(inputs_embeds.shape)) == 2: inputs_embeds = inputs_embeds.unsqueeze(0)

        # the shape of the tensor that will be returned will be: [batch_size, max_sequence_length, number_embeddings] -> [8, 600, 512]
        # learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        learned_embeds = self.soft_prompt.repeat(inputs_embeds.size(0), 1, 1)

        # print('shape learned embeds: ' + str(learned_embeds.shape))

        # inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)
        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        # print('shape inputs embeds: ' + str(inputs_embeds.shape))
        return inputs_embeds

    # to make sure that padding token ids of the labels are not taken into account by the loss function
    # this method extends the labels tensor by elements that are ignored by the CrossEntropyLoss function
    # this can be done using the ignore_index value -100
    def extend_labels(self, labels, ignore_index=-100):
        if len(list(labels.shape)) == 1: labels = labels.unsqueeze(0)
        number_of_batches = labels.shape[0]

        # print('number batches: ' + str(n_batches))

        # return a new tensor of shape [number_of_batches, number_tokens+labels] that is filled with the ignore_index value (-100)
        return torch.cat([torch.full((number_of_batches, self.number_tokens), ignore_index).to(self.device), labels],
                         dim=1)

    def extend_attention_mask(self, attention_mask):
        # prepend a new dimension (1) to the shape of attention_mask in case it is one dimensional
        if len(list(attention_mask.shape)) == 1: attention_mask = attention_mask.unsqueeze(0)

        # get the number of batches
        number_of_batches = attention_mask.shape[0]

        # return a new tensor of shape [number_of_batches, number_tokens+attention_mask] that is filled with the ones
        return torch.cat([torch.full((number_of_batches, self.number_tokens), 1).to(self.device), attention_mask],
                         dim=1)

    def save_soft_prompt(self, filename: str = "soft_prompt.model"):
        torch.save(self.soft_prompt, 't5-tuning/soft_prompts/' + filename)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # This fixes CUDA for some reason
        # print(kwargs)
        # kwargs['input_ids'] = kwargs['input_ids'].to(self.device)
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        return super().prepare_inputs_for_generation(input_ids, None, *args, **kwargs)

    def forward(
            self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
            encoder_attention_mask=None, labels=None, use_cache=None, output_attentions=None,
            output_hidden_states=None, return_dict=None, decoder_input_ids=None, encoder_outputs=None,
            decoder_head_mask=None, cross_attn_head_mask=None):

        # print(input_ids)
        if input_ids is not None:
            inputs_embeds = self.append_learned_embedding_to_input(input_ids)
            print("1")

        if labels is not None:
            labels = self.extend_labels(labels)
            print("2")

        if attention_mask is not None:
            attention_mask = self.extend_attention_mask(attention_mask)
            # print("3")

        return super().forward(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict
        )

epochs = 10

# Check GPU availability
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

torch.cuda.empty_cache()


# Prompt-tuning
# number of prompt tokens
number_prompt_tokens = 50

# If set to true, the soft prompt will be initialized from the models vocabulary
# Otherwise, it will be randomly (uniformly in a range) initialized.
random_range = 0.5
init_from_vocab = True

#torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)

tokenizer_t5_small = T5Tokenizer.from_pretrained('t5-small')

# Instantiate one T5 small model that should be trained on all the 3 datasets
model_t5_small = T5PromptTuning.from_pretrained('t5-small', number_tokens=number_prompt_tokens, initialize_from_vocab=init_from_vocab)

#moving the models to device(GPU/CPU)
model_t5_small.to(dev)

load_no_duplicate_sets = True
train_data_amr = pd.read_csv('data/amr/train/abstract_meaning_representation_train.csv' if not load_no_duplicate_sets else 'data/amr/train/amr_train_no_duplicate_inputs.csv')
test_data_amr = pd.read_csv('data/amr/test/abstract_meaning_representation_test.csv' if not load_no_duplicate_sets else 'data/amr/test/amr_test_no_duplicate_inputs.csv')

train_data_amr = train_data_amr.sort_values(by='input_text', ignore_index=True)
test_data_amr = test_data_amr.sort_values(by='input_text', ignore_index=True)

# Trimming off the last few datapoints from AMR so that a batch would not leave any remainder.
train_data_amr = train_data_amr.iloc[:len(train_data_amr)-4,:] if not load_no_duplicate_sets else train_data_amr.iloc[:len(train_data_amr)-6,:]
test_data_amr = test_data_amr.iloc[:len(test_data_amr)-6,:] if not load_no_duplicate_sets else test_data_amr.iloc[:len(test_data_amr)-3,:]

batch_size_amr = 8
number_of_batches_train_amr = int(len(train_data_amr)/batch_size_amr)
number_of_batches_test_amr = int(len(test_data_amr)/batch_size_amr)
print('--- Number of train batches AMR : ' + str(number_of_batches_train_amr) + ' --- ')
print('--- Number of test  batches AMR : ' + str(number_of_batches_test_amr) + '  --- ')

inputs_test_amr, \
labels_test_amr = create_list_of_batches(batch_size=batch_size_amr,
                                              num_batches=number_of_batches_test_amr,
                                              data=test_data_amr,
                                              tokenizer=tokenizer_t5_small)

optimizer_t5 = optimizer_adafactor(model_t5_small)

def make_predictions(model, inputs_test, tokenizer, challenge_name):

  model_predictions = []
  model.eval()
  with torch.no_grad():
    for i in range(len(inputs_test)):
      print(i)
      output = tokenizer.batch_decode(model.generate(#input_ids=inputs_test[i]['input_ids'],
                                                     do_sample=True,
                                                     max_length=400,
                                                     top_p=0.92,
                                                     top_k=0,
                                                     bos_token_id=0,
                                                     pad_token_id=0,
                                                     eos_token_id=1,
                                                     inputs_embeds=model.append_learned_embedding_to_input(inputs_test[i]['input_ids']),
                                                     attention_mask=model.extend_attention_mask(inputs_test[i]['attention_mask'])
                                                     ),
                                        skip_special_tokens=True)

      model_predictions.append([x.replace('<pad>','').replace('</s>','').strip() for x in output])

    # flatten the predictions list which has the length of batch_size * number_of_batches
    model_predictions = list(chain(*model_predictions))
  model.train()
  with open('drive/MyDrive/MIwDL/data/' + challenge_name + '/test/prompt_tuning_hypothesis/hypothesis', 'w') as file:
    for i in range(len(model_predictions)):
      file.write(model_predictions[i] + '\n' if i < len(model_predictions)-1 else model_predictions[i])
  return model_predictions

model_predictions = make_predictions(model=model_t5_small,
                         inputs_test=inputs_test_amr,
                         tokenizer=tokenizer_t5_small,
                         challenge_name='amr')