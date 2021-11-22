# Pre-Training

## Text-to-Text Pre-Training for Data-to-Text Tasks (Kale and Rastogi, 2021, arXiv:2005.10433v3)
- paper goal: pre-train + fine-tune strategy   for data-to-text tasks
- paper main finding: enables simple, end-to-end transformer based models to outperform pipelined neural architectures tailored for data-to-text generation

- Why pretraining?
  - improves robustness of models to out-of-domain inputs

- Related techniques:
  - transfer learning

- Their approach:
  - first: select pretrained models from T5:
    - small, 60 million parameters
    - base, 220 million parameters
    - large, 770 million parameters
    - 3B, 3 billion parameters
  - second: fine-tuning of dedicated models for dedicated downstream tasks
    - they use three datasets -> 3x model fine-tuning
  - For each dataset (downstream task) they trained multiple models:
    - WebNLG: four model architectures
    - ToTTo: two model architectures
    - MultiWoz: two model architectures

- Datasets they used:
  - ToTTo: Wikipedia tables paired with natural language descriptions
  - MultiWoz: 10k human-human dialogs
    - can also be used for NLG tasks
  - WebNLG: graph of subject-object-predicate triples -> have to be converted into a textual description
    - we also will use this one
  - each of the datasets uses different kind of structured data

- Evaluation
  - used benchmarks:
    - BLEU
    - METEOR
    - PARENT
    - SER
  - human evaluation:
    - 3 humans evaluated the outcomings
  
- Impact of model capacity:
  - their results suggest a dependency of performance on the size and complexity of the dataset
  - 