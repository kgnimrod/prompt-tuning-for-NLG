from datetime import datetime
from os.path import join

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import src.core.pre_process as pre_process
from src.core.inference import make_predictions
from src.core.t5_promt_tuning import T5PromptTuning
from src.core.config import load_config_from_yaml
from src.core.persistance import save_soft_prompt, load_model, save_state_dict, validate_path
from src.core.train import train


class Experiment:
    def __init__(self, config):
        self.count = 0
        self.config = config
        self.number_prompt_tokens = self.config["NUMBER_PROMPT_TOKENS"]
        self.random_range = self.config["RANDOM_RANGE"]
        self.init_from_vocab = self.config["INIT_FROM_VOCAB"]

        datasets = self.config["DATASETS"]
        self.source_datasets = {}
        for dataset in datasets:
            self._load_dataset(dataset)

        if self.config["SAMPLE"]:
            self.data = pre_process.sample(
                self.source_datasets,
                self.config["SAMPLE_SIZE_TRAIN"],
                self.config["SAMPLE_SIZE_EVAL"]
            )
        else:
            self.data = pre_process.combine(self.source_datasets)

        self.tokenizer = None
        self.model = None
        self.predictions = None
        self.inputs = {}
        self.starting_timestamp = datetime.timestamp(datetime.now())

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
            'output_dir':  join("runs", self.config["OUTPUT_DIR"]),
            'prediction_loss_only': self.config["PREDICTION_LOSS_ONLY"],
            'remove_unused_columns': self.config["REMOVE_UNUSED_COLUMNS"],
            'save_model': self.config["SAVE_MODEL"],
            'save_steps': self.config["SAVE_STEPS"],
            'save_total_limit': self.config["SAVE_TOTAL_LIMIT"],
            'wandb_run_name': self.config["WANDB_RUN_NAME"],
            'starting_timestamp': str(self.starting_timestamp),
            'train_with_embeddings': self.config["PROMPT_TUNING"]
        }

    def _load_dataset(self, dataset):
        dataset_config = load_config_from_yaml(dataset["DATASET_CONFIG"])

        if dataset_config["PRE_PROCESS"] == 'custom':
            self.source_datasets[dataset["KEY"]] = getattr(pre_process, dataset_config["PRE_PROCESS_METHOD"])(dataset_config)
        else:
            self.source_datasets[dataset["KEY"]] = pre_process.pre_process_huggingface_dataset(dataset_config)

    def run(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.config["PRE_TRAINED_MODEL"])
        self._set_model()

        if self.config["TRAIN"]:
            self.inputs['train'] = self._prepare_inputs(self.data['train'])
            self.inputs['validation'] = self._prepare_inputs(self.data['validation'])
        if self.config["EVALUATE"]:
            self.inputs['test'] = self._prepare_inputs(self.data['test'])
        self._to_device()
        if self.config["TRAIN"]:
            train(self.training_args, self.model, self.inputs, self.config["TRAIN_MODE"])

        if self.config["SAVE_MODEL"]:
            validate_path(self.training_args["output_dir"])
            validate_path(join(self.training_args["output_dir"], "models"))
            save_state_dict(
                self.model,
                join(self.training_args["output_dir"], "models"),
                "model_state_dict_started_" + str(self.starting_timestamp)
            )

        if self.config["SAVE_SOFT_PROMPTS"]:
            validate_path(self.training_args["output_dir"])
            validate_path(join(self.training_args["output_dir"], "models"))
            save_soft_prompt(
                self.model,
                join(self.training_args["output_dir"], "models"),
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
        print(device)
        self.model.to(device)
        # self.model = torch.nn.DataParallel(model, device_ids=config["GPUS"])
        for item in self.inputs:
            self.inputs[item]["input_ids"].to(device)
            self.inputs[item]["attention_mask"].to(device)
            self.inputs[item]["labels"].to(device)
        torch.cuda.empty_cache()
