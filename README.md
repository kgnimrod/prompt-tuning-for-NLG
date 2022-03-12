# Prompt-Tuning-for-NLG
Repository for Prompt Tuning for Natural Language Generation tasks.

There are two ways how to run and experiment with our code basis: Via the jupyter notebooks or on the cluster using docker images

## 1. Jupyter Notebooks
To easily get a detailed impression on how our implementation and experimentation looks like, we provide three notebook files for three cases: 
- model fine-tuning in: 'fine_tuning_t5.ipynb'
- single soft prompt tuning in: 'single_soft_prompt_tuning_t5.ipynb' and
- mixed task soft prompt tuning in: 'mixed_task_soft_prompt_tuning_t5.ipynb'

The notebooks can be easily used to run a whole training and inference iteration with different, configurable (hyperparameter-)settings.
We used these notebooks primarily on Google Colab (Kaggle also works) to be able to access GPU resources. 

## 2. Running experiments on the Cluster via Docker Images

To build the docker image run the following code snippet in this projects main folder:
```
docker image build -t midl_nlg:test .
```

To run a container run the following command:
```
docker container run --rm --volume="$(pwd):/opt/workspace" midl_nlg:test
```

You can built an enroot image and run containers via enroot:
```
enroot create --name <midl_nlg_test> <name_of_your_eonroot_image.sqsh>
enroot list -f
enroot start --rw --mount .:/opt/workspace <midl_nlg_test>
```

Or manually run an experiment via python:
```
python main.py --experiment <name_of_experiment.yml>
```
The experiment-yaml-file is expected to be found in the experiments folder of this project.

## Calculation of Scores using the official evaluation scripts
We cloned the official evaluation scripts from the respective repositories:
- E2E NLG: https://github.com/tuetschek/e2e-metrics
- Web NLG 2020: https://github.com/WebNLG/GenerationEval

You can find the corresponding README files on how to use these scripts:
- E2E NLG: [here](official_evaluation/e2e_evaluation/README.md) and 
- Web NLG: [here](official_evaluation/web_nlg_amr_evaluation/README.md)

After finishing the inference phase, i.e. the output texts are generated and stored in a file 'hypothesis', please store this
file in the corresponding subdirectory data/<your_challenge>/test/<'fine_tuning_hypothesis' or 'prompt_tuning_hypothesis'>/hypothesis

Then, to run the scripts, please execute the following terminal command from the corresponding subdirectories:
- for AMR and Web NLG 2020 evaluations:
  - open `eval.py` and check whether lines `324` and `325` reference the correct challenge path (due to an external bug from the script authors, unfortunately the paths cannot be passed as terminal arguments and so need to be changed in the file)
  - Then, just execute: `python eval.py` from ***official_evaluation/web_nlg_amr_evaluation/GenerationEval-master***
    
- for E2E evaluations:
 - Execute: `python measure_scores.py -p -t -H ../../../data/e2e/test/reference ../../../data/e2e/test/prompt_tuning_hypothesis/hypothesis` from ***official_evaluation/e2e_evaluation/e2e-metrics-master***