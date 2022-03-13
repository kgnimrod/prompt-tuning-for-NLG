from datetime import datetime
from os.path import join

from torch.utils.data import DataLoader
# import wandb

from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, IntervalStrategy

import src.core.pre_process as pre_process
from src.core.evaluation import predict, compute_scores
from src.core.t5_promt_tuning import T5PromptTuningLM, T5PromptTuningEmbeddings
from src.core.config import load_config_from_yaml
from src.core.persistance import load_model, save_state_dict, validate_path, save_predictions, \
    load_state_dict, save_scores


class Experiment:
    def __init__(self, config):
        self.config = config
        self.starting_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.path = self._validate_paths()
        if self.config["PROMPT_TUNING"]:
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
        self.scores = None
        self.inputs = {}

        # if "wandb" == self.config["REPORT_TO"]:
        #     wandb_config = load_config_from_yaml(self.config["WANDB_CONFIG"])
        #     wandb.init(project=wandb_config["WANDB_PROJECT"], entity=wandb_config["WANDB_ENTITY"])
        self.trainer_args = self._load_trainer_args()

    def run(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.config["PRE_TRAINED_MODEL"])
        self._load_model()

        if self.config["TRAIN"]:
            self.inputs['train'] = self._prepare_inputs(self.data['train'])
            self.inputs['validation'] = self._prepare_inputs(self.data['validation'])

            self._train()

            if self.config["SAVE_MODEL"]:
                save_state_dict(
                    self.model,
                    join(self.path, "models")
                )

        if self.config["EVALUATE"]:
            self.inputs['test'] = self._prepare_inputs(self.data['test'])

            self._predict()
            save_predictions(
                self.predictions["predictions"],
                join(self.path, "predictions")
            )
            self._evaluate()
            save_scores(self.scores, join(self.path, "scores"))

    def _load_dataset(self, dataset):
        dataset_config = load_config_from_yaml(dataset["DATASET_CONFIG"])

        if dataset_config["PRE_PROCESS"] == 'custom':
            self.source_datasets[dataset["KEY"]] = getattr(pre_process, dataset_config["PRE_PROCESS_METHOD"])(
                dataset_config)
        else:
            self.source_datasets[dataset["KEY"]] = pre_process.pre_process_huggingface_dataset(dataset_config)

    def _load_trainer_args(self):
        return TrainingArguments(
            evaluation_strategy=IntervalStrategy.EPOCH,
            greater_is_better=self.config["GREATER_IS_BETTER"],
            learning_rate=self.config["LEARNING_RATE"],
            load_best_model_at_end=self.config["LOAD_BEST_MODEL_AT_END"],
            logging_first_step=self.config["LOGGING_FIRST_STEP"],
            logging_steps=self.config["LOGGING_STEPS"],
            metric_for_best_model=self.config["METRIC_FOR_BEST_MODEL"],
            num_train_epochs=self.config["NUM_TRAIN_EPOCHS"],
            optim=self.config["OPTIMIZER"],
            lr_scheduler_type=self.config["LR_SCHEDULER"],
            output_dir=join(self.path, "logs"),
            per_device_train_batch_size=self.config["BATCH_SIZE"],
            per_device_eval_batch_size=self.config["EVAL_BATCH_SIZE"],
            prediction_loss_only=self.config["PREDICTION_LOSS_ONLY"],
            remove_unused_columns=self.config["REMOVE_UNUSED_COLUMNS"],
            report_to=self.config["REPORT_TO"],
            run_name=self.config["WANDB_RUN_NAME"],
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=self.config["SAVE_TOTAL_LIMIT"]
        )

    def _load_model(self):
        if self.config["PROMPT_TUNING"]:
            self.model = T5PromptTuningLM.from_pretrained(
                self.config["PRE_TRAINED_MODEL"],
                number_tokens=self.number_prompt_tokens,
                initialize_from_vocab=self.init_from_vocab
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config["PRE_TRAINED_MODEL"]
            )

        if self.config["LOAD_MODEL"]:
            if self.config['TRAIN']:
                self.model = load_model(self.config["INPUT_DIR"])
            else:
                load_state_dict(self.model, self.config["INPUT_DIR"], self.config["INPUT_FILE"])

    def _prepare_inputs(self, dataset):
        # train_dataset.shard(num_shards=4, index=0)
        dataset = dataset.map(self._tokenize, batched=True, batch_size=self.config["BATCH_SIZE"])

        for feature in dataset.features:
            if feature not in ['input_ids', 'attention_mask', 'labels']:
                dataset = dataset.remove_columns(feature)

        dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'])
        return dataset

    def _tokenize(self, batch):
        inputs = []
        for item in batch['input_ids']:
            inputs.append('Translate from Graph to Text: ' + item)

        tokenized_input = self.tokenizer.batch_encode_plus(
            inputs, padding='max_length', max_length=500
        )

        labels = []
        for item in batch['labels']:
            labels.append(item)

        tokenized_labels = self.tokenizer.batch_encode_plus(
            labels, padding='max_length', max_length=500
        )
        tokenized_input['labels'] = tokenized_labels['input_ids']

        return tokenized_input

    def _train(self):
        trainer = Trainer(
            model=self.model,
            args=self.trainer_args,
            train_dataset=self.inputs["train"],
            eval_dataset=self.inputs["validation"],
            # compute_metrics=compute_metrics
        )
        trainer.train()

    def _predict(self):
        print("start decoding")
        embeddings = None
        model = self.model
        if self.config["PROMPT_TUNING"]:
            embeddings = T5PromptTuningEmbeddings(self.model)
            model = self.model
            # model = self.model = T5ForConditionalGeneration.from_pretrained(
            #     self.config["PRE_TRAINED_MODEL"]
            # )

        # val_loader = DataLoader(dataset=self.inputs["test"], batch_size=8, num_workers=0)
        self.predictions = predict(self.tokenizer, model, self.inputs["test"], embeddings)
        print("finished decoding")

    def _evaluate(self):
        self.scores = compute_scores(self.predictions["predictions"], self.predictions["targets"])

    def _validate_paths(self):
        path = validate_path(join("runs"))
        path = validate_path(join(path, self.starting_time + "_" + self.config["OUTPUT_DIR"]))
        validate_path(join(path, "models"))
        validate_path(join(path, "logs"))
        validate_path(join(path, "predictions"))
        validate_path(join(path, "scores"))
        return path
