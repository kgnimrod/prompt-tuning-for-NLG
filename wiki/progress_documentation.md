# Progress Documentation for Promp Tuning for NLG

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

- See [T5 Documentation](t5_documentation.md) for more information on how T5 works, details of the model, which downstream tasks were considered and on which dataset it was trained on