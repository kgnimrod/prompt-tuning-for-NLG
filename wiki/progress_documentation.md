# Progress Documentation for Promp Tuning for NLG

## Ongoing literature research
- [T5 Documentation](literature/t5.md) for more information on how T5 works, details of the model, which downstream tasks were considered and on which dataset it was trained on
- [Attention and Transformers](literature/attention_and_transformers.md)
- [Text-to-Text Pre-Trainings](literature/pretraining.md)

## 08. & 10.11.2021:
- Kick-Off and project scope framing together with teaching team
- (Private) Github Repository created: https://github.com/caesarea38/prompt-tuning-for-NLG
- Invited teaching team to join the repo
- Papers read:
  - _Text-to-text Pre-Training for Data-to-Text Tasks_ (https://arxiv.org/abs/2005.10433)
  - _PTR: Promp Tuning with Rules for Text Classification_ (https://arxiv.org/abs/2105.11259)

## 15.11.2021 & 17.11.2021:
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
```
Output:
> "Der Tisch ist wunderbar."


## 22. 11. 2021
- Clarification project resources usage
- remark: metrics and benchmarks:
  - we can use same once as mentioned in the different papers > we decide which onces we want to use
- question: what is benefit of self-attention over other techniques
- question: masking vs. fully visible input into the encoder
- remark: first delivery: 
  - what is our project goal
  - what is our motivation
  - what is the techniques and tools we'd like to use
  - what is our plan / roadmap
  - from the point of a report that would cover the sections: goal, motivation, state-of-art
- remark: documentation:
  - we could also add section for shared slides set
  - transition from just reading and understanding to what we actually want to incorporate into project
    - already add parts we want to use into those slides
    - refine the parts, e.g. add graphics, combine different resources
  - with this we can work on a draft for first presentation continuously