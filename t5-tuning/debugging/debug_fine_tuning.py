
import pandas as pd
import os
import torch
import time
import warnings
import matplotlib.pyplot as plt
from itertools import chain
warnings.filterwarnings('ignore')

from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdamW
from IPython.display import HTML, display

def create_list_of_batches(batch_size, num_batches, data, tokenizer):
# Create List of batches for inputs and labels
    inputs = []
    labels = []
    for i in range(num_batches):
        input_batch=[]
        label_batch=[]
        for index,row in data[i*batch_size:i*batch_size+batch_size].iterrows():
          input_batch.append('translate from Graph to Text: '+row['input_text']+'</s>')
          label_batch.append(row['target_text']+'</s>')

        input_batch=tokenizer.batch_encode_plus(input_batch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
        label_batch=tokenizer.batch_encode_plus(label_batch,padding=True,max_length=400,return_tensors='pt')["input_ids"]

        input_batch=input_batch.to(dev)
        label_batch=label_batch.to(dev)

        inputs.append(input_batch)
        labels.append(label_batch)
    return inputs, labels


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
        model.parameters(),
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

epochs = 10

# Check GPU availability
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')

model_t5_small_amr = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
model_t5_small_amr.to(dev)

load_no_duplicate_sets = True
# Load the datasets for the Abstract Meaning Representation AMR challenge
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

inputs_train_amr, labels_train_amr = create_list_of_batches(batch_size=batch_size_amr,
                                                                    num_batches=number_of_batches_train_amr,
                                                                    data=train_data_amr,
                                                                    tokenizer=tokenizer_t5)

inputs_test_amr, labels_test_amr = create_list_of_batches(batch_size=batch_size_amr,
                                                                    num_batches=number_of_batches_test_amr,
                                                                    data=test_data_amr,
                                                                    tokenizer=tokenizer_t5)

optimizer_t5_amr = optimizer_adafactor(model_t5_small_amr)


# Train T5 small on AMR
"""model_t5_small_amr = trainer(model=model_t5_small_amr,
                         num_batches=number_of_batches_train_amr,
                         inputs=inputs_train_amr,
                         labels=labels_train_amr,
                         optimizer=optimizer_t5_amr,
                         challenge_name='AMR',
                         model_name='base')"""

