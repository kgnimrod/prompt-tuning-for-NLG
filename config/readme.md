# What is this Folder about?
This folder contains configuration files for the different
datasets used in this project. These files encapsulate reusable
parts of multiple experiments.

# How to modify config files or create your own?
For the given downstream tasks WEB NLG 2020, E2E NLG und LDC2020T02, we provide
ready to use configurations. When create your own configuration
please provide the following parameters:

* DATASET_PATH: put the path to the dataset here. 
In case you have a custom dataset put the file path here.
In case you are using huggingface datasets put the value of the
path parameter of the huggingface dataset here.
* DATASET_NAME: put the value of the huggingface datasets name
parameter here. In case you have a custom dataset use:
  * DATASET_NAME_TRAIN
  * DATASET_NAME_TEST

* NAME_TRAIN_DATASET: name of the training data subset
* NAME_VALIDATION_DATASET: name of the validation data subset
* NAME_TEST_DATASET: name of the test data subset

* PRE_PROCESS: (huggingface / custom)
  * huggingface: in case you use huggingface datasets
  * custom: in case you use a custom dataset
* PRE_PROCESS_METHOD: the python method that processes the data.
This parameter is only necessary for custom datasets. 
The method has to exist and be implemented in the file
`src_cluster/core/pre_process.py`

* FLATTEN: boolean parameter to indicate if the input dataset
needs to be flattened.
* INPUT_IDS: column names of the input data
* LABELS: column names of the labels

# Special cases:wandb
This project supports wandb integration. To use it please create
a file name `wandb.yml`.  (We put this file on our gitignore list.) 
Please provide the following parameters:

* WANDB_PROJECT: name of the project on wandb
* WANDB_ENTITY: name of your wandb entity