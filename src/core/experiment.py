from datetime import datetime
from os import mkdir
from os.path import join, exists

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import src.core.pre_process as pre_process
from src.core.inference import make_predictions
from src.core.t5_promt_tuning import T5PromptTuning
from src.core.config import load_config_from_yaml
from src.core.persistance import save_soft_prompt, save_model, load_model
from src.core.train import train


class Experiment:
    def __init__(self, config):
        self.count = 0
        self.config = config
        self.dataset_config = load_config_from_yaml(self.config["DATASET_CONFIG"])
        self.number_prompt_tokens = self.config["NUMBER_PROMPT_TOKENS"]
        self.random_range = self.config["RANDOM_RANGE"]
        self.init_from_vocab = self.config["INIT_FROM_VOCAB"]

        if self.dataset_config["PRE_PROCESS"] == 'custom':
            self.datasets = getattr(pre_process, self.dataset_config["PRE_PROCESS_METHOD"])(self.dataset_config)
        else:
            self.datasets = pre_process.pre_process_huggingface_dataset(self.dataset_config)

        self.tokenizer = None
        self.model = None
        self.predictions = None
        self.inputs = {}
        self.starting_timestamp = datetime.timestamp()


        self.training_args = {
            'batch_size': self.config["BATCH_SIZE"],
            'eval_batch_size': self.config["EVAL_BATCH_SIZE"],
            'eval_accumulation_steps': self.config["EVAL_ACCUMULATION_STEPS"],
            'eval_steps': self.config["EVAL_STEPS"],
            'greater_is_better': self.config["GREATER_IS_BETTER"],
            'learning_rate': self.config["LEARNING_RATE"],
            'load_best_model_at_end': self.config["LOAD_BEST_MODEL_AT_END"],
            'logging_first_step': self.config["LOGGING_FIRST_STEP"],
            'logging_steps': self.config["LOGGING_STEPS"],
            'metric_for_best_model': self.config["METRIC_FOR_BEST_MODEL"],
            'num_train_epochs': self.config["NUM_TRAIN_EPOCHS"],
            'output_dir':  join("runs", self.config["OUTPUT_DIR"] + "_" + str(self.starting_timestamp)),
            'prediction_loss_only': self.config["PREDICTION_LOSS_ONLY"],
            'remove_unused_columns': self.config["REMOVE_UNUSED_COLUMNS"],
            'save_model': self.config["SAVE_MODEL"],
            'save_steps': self.config["SAVE_STEPS"],
            'save_total_limit': self.config["SAVE_TOTAL_LIMIT"],
            'wandb_run_name': self.config["WANDB_RUN_NAME"]
        }

    def run(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.config["PRE_TRAINED_MODEL"])
        self._set_model()

        if self.config["TRAIN"]:
            self.inputs['train'] = self._prepare_inputs(self.datasets['train'])
            self.inputs['validation'] = self._prepare_inputs(self.datasets['validation'])
        if self.config["EVALUATE"]:
            self.inputs['test'] = self._prepare_inputs(self.datasets['test'])
        self._to_device()
        if self.config["TRAIN"]:
            train(self.training_args, self.model, self.inputs)

        if self.config["SAVE_MODEL"]:
            save_model(self.model, join(self.training_args["output_dir"], "models"))

        if self.config["SAVE_SOFT_PROMPTS"]:
            save_soft_prompt(
                self.model,
                join(self.training_args["output_dir"], "models"),
                self.training_args["output_dir"],
                self.training_args["num_train_epochs"],
                self.config["PRE_TRAINED_MODEL"],
                self.number_prompt_tokens
            )

        if self.config["EVALUATE"]:
            self.predictions = make_predictions(
                self.model, self.inputs['test'], self.tokenizer, use_embeddings=self.config["PROMPT_TUNING"]
            )

    def _set_model(self):
        if self.config["LOAD_MODEL"]:
            self.model = load_model(self.config["INPUT_DIR"])
        else:
            if self.config["PROMPT_TUNING"]:
                self.model = T5PromptTuning.from_pretrained(
                    self.config["PRE_TRAINED_MODEL"],
                    number_tokens=self.number_prompt_tokens,
                    initialize_from_vocab=self.init_from_vocab
                )
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config["PRE_TRAINED_MODEL"]
                )

    def _prepare_inputs(self, dataset):
        # train_dataset.shard(num_shards=4, index=0)
        dataset = dataset.map(self._tokenize, batched=True, batch_size=self.training_args["batch_size"])

        for feature in dataset.features:
            if feature not in ['input_ids', 'attention_mask', 'labels']:
                dataset = dataset.remove_columns(feature)

        dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'])
        return dataset

    def _tokenize(self, batch):
        inputs = []
        for item in batch['input_ids']:
            inputs.append(item[0][0])

        tokenized_input = self.tokenizer.batch_encode_plus(
            inputs, padding='max_length', max_length=500
        )

        labels = []
        for item in batch['labels']:
            if len(item) > 0:
                labels.append(item[0])
            else:
                labels.append('None')
                self.count += 1

        tokenized_labels = self.tokenizer.batch_encode_plus(
            labels, padding='max_length', max_length=500
        )
        tokenized_input['labels'] = tokenized_labels['input_ids']

        return tokenized_input

    def _to_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # self.model = torch.nn.DataParallel(model, device_ids=config["GPUS"])
        for item in self.inputs:
            self.inputs[item]["input_ids"].to(device)
            self.inputs[item]["attention_mask"].to(device)
            self.inputs[item]["labels"].to(device)
        torch.cuda.empty_cache()
