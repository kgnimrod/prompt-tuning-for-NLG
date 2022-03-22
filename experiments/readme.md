# What is this Folder about?
This folder contains configuration files for the different
experiments conducted in this project. These files encapsulate all
information to instantiate multiple runs based on the same underlying
experiment.

# How to modify config files or create your own?
In this project we worked with the following three datasets:

* WEB NLG 2020
* E2E NLG
* LDC2020T02, we provide

and the pretrained models t5-small and t5-base of the huggingface 
transformers package.

The experiments make use of the datasets config files.
The configurable parameters are parameters:

* DATASETS: list of datasets -> multiple datasets can be used in
one experiment
  * KEY: identifier for the dataset
  * DATASET_CONFIG: path to the dataset config file 
  (recommended putting in this project's config folder)

* PRE_TRAINED_MODEL: name of the huggingface pre-trained model
* PROMPT_TUNING: boolean, indicates if prompt tuning is used
(or fine-tuning)
* LOAD_MODEL: boolean, indicates if a given model should be loaded
* SAVE_MODEL: boolean, indicates if the model should be saved 
after training

* SAMPLE: boolean, indicates if the data should be sampled
* SAMPLE_SIZE_TRAIN: number of items per dataset
* SAMPLE_SIZE_EVAL: number of items per dataset

* NUMBER_PROMPT_TOKENS: 50
* RANDOM_RANGE: 0.5
* INIT_FROM_VOCAB: True

* TRAIN: boolean, indicates if training step should be performed
* EVALUATE: boolean, indicates if evaluation (prediction + scoring)
step should be performed

* supported huggingface trainer parameters:
  * BATCH_SIZE
  * EVAL_BATCH_SIZE
  * GREATER_IS_BETTER
  * LEARNING_RATE
  * LOAD_BEST_MODEL_AT_END
  * LOGGING_FIRST_STEP
  * LOGGING_STEPS
  * METRIC_FOR_BEST_MODEL
  * NUM_TRAIN_EPOCHS
  * OPTIMIZER
  * LR_SCHEDULER
  * PREDICTION_LOSS_ONLY
  * REMOVE_UNUSED_COLUMNS
  * REPORT_TO
  * SAVE_TOTAL_LIMIT

* parameters for wandb integration:
  * WANDB_RUN_NAME:name of the wandb run
  * WANDB_CONFIG: path + filename to the wandb config file 
  (recommended being in the config subfolder and named wandb.yml)

* OUTPUT_DIR: folder for output files
  * the project will automatically create a "runs" folder and a
subfolder for this experiment with the given output_dir name
* INPUT_DIR: folder name where to load a model from
  * the project will expect the model to be in a subfolder of this
  folder, named "models"
