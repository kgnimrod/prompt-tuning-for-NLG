#Progress Documentation for Promp Tuning for NLG

## 10.11.2021:
- (Private) Github Repository created: https://github.com/caesarea38/prompt-tuning-for-NLG
- Invited Haojin, Ting still missing
- Papers read: 
  - _Text-to-text Pre-Training for Data-to-Text Tasks_ (https://arxiv.org/abs/2005.10433)
  - _PTR: Promp Tuning with Rules for Text Classification_ (https://arxiv.org/abs/2105.11259)

## 14.11.2021 & 17.11.2021:
- Added the huggingface transformers module as submodule to our repository
- Created a directory `playground` to do testing and experimenting with the API 
- Created a jupyter notebook file `huggin_face_first_steps.ipynb` to go through different examples on how to use the pre-trained pipelines that are provided per NLP task
- Example for the pre-trained Model T5 (Machine Translation):

```python
from transformers import pipeline

# generate tokenizer object from pre-trained model T5
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# instantiate model object
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer('translate English to German: The table is wonderful.', return_tensors='pt').input_ids

# compute outputs and print the resulting translation
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

>>> "Der Tisch ist wunderbar."
```
- Some important points from the main T5 paper: _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_ by Raffel et al. (2020), https://arxiv.org/pdf/1910.10683.pdf
    - T5 = "Text-to-Text Transfer Transformer"
    - Based on Transfer Learning as main approach, i.e. first pre-train a model on a data rich task and then fine tune it on a downstream task
    - Idea underlying the main work: Treat every text processing problem as a "text-to-text" problem, i.e. taking text as input and producing new text as output
    - -> Allows to compare the effectiveness of different transfer learning objectives
    - Downstream Tasks benchmarked in this paper: Sentiment Analysis, Natural Language Inference, Sentence Completion, Question answering etc.
    - model architecture is based on a standard encoder-decoder Transformer as proposed by Vaswani et al. (2017)
    - Pre-training: 524.288 training steps per model on C4 dataset before fine-tuning
    - Fine-Tuning: 262.144 steps on all task.
    - Denoising objective or Masked Language Modeling used to predict missing or otherwise corrupted tokens in the input.
    - "Word dropout" regularization technique: randomly sample and drop out X percent of tokens in the input sequence.  
    - <img src="/Users/furkansimsek/Desktop/Master_HPI/2.Semester/MachineIntelligenceWithDeepLearning/prompt-tuning-for-NLG/wiki/images/objective_schema_t5.png" width="900" height="400"/>
