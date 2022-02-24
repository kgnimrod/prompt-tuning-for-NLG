from datetime import datetime
from os.path import join

import torch
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, IntervalStrategy

import src.core.pre_process as pre_process
from src.core.t5_promt_tuning import T5PromptTuningLM
from src.core.config import load_config_from_yaml
from src.core.persistance import load_model, save_state_dict, validate_path, save_predictions, \
    load_state_dict


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

        self.trainer_args = self._load_trainer_args()

    def run(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.config["PRE_TRAINED_MODEL"])
        self._load_model()

        if self.config["TRAIN"]:
            self.inputs['train'] = self._prepare_inputs(self.data['train'])
            self.inputs['validation'] = self._prepare_inputs(self.data['validation'])
        if self.config["EVALUATE"]:
            self.inputs['test'] = self._prepare_inputs(self.data['test'])

        if self.config["TRAIN"]:
            self._train()

            if self.config["SAVE_MODEL"]:
                path = join("runs", self.config["OUTPUT_DIR"])
                path = validate_path(path)
                path = validate_path(join(path, "models"))
                save_state_dict(
                    self.model,
                    path,
                    "model_state_dict_started_" + str(self.starting_timestamp)
                )

        if self.config["EVALUATE"]:
            self._predict()
            path = join("runs", self.config["OUTPUT_DIR"])
            path = validate_path(path)
            path = validate_path(join(path, "predictions"))
            save_predictions(self.predictions, path, "predictions_" + str(self.starting_timestamp))

    def _load_dataset(self, dataset):
        dataset_config = load_config_from_yaml(dataset["DATASET_CONFIG"])

        if dataset_config["PRE_PROCESS"] == 'custom':
            self.source_datasets[dataset["KEY"]] = getattr(pre_process, dataset_config["PRE_PROCESS_METHOD"])(
                dataset_config)
        else:
            self.source_datasets[dataset["KEY"]] = pre_process.pre_process_huggingface_dataset(dataset_config)

    def _load_trainer_args(self):
        return TrainingArguments(
            eval_accumulation_steps=self.config["EVAL_ACCUMULATION_STEPS"],
            eval_steps=self.config["EVAL_STEPS"],
            evaluation_strategy=IntervalStrategy.STEPS,
            greater_is_better=self.config["GREATER_IS_BETTER"],
            learning_rate=self.config["LEARNING_RATE"],
            load_best_model_at_end=self.config["LOAD_BEST_MODEL_AT_END"],
            logging_first_step=self.config["LOGGING_FIRST_STEP"],
            logging_steps=self.config["LOGGING_STEPS"],
            metric_for_best_model=self.config["METRIC_FOR_BEST_MODEL"],
            num_train_epochs=self.config["NUM_TRAIN_EPOCHS"],
            output_dir=join("runs", self.config["OUTPUT_DIR"], "logs"),
            per_device_train_batch_size=self.config["BATCH_SIZE"],
            per_device_eval_batch_size=self.config["EVAL_BATCH_SIZE"],
            prediction_loss_only=self.config["PREDICTION_LOSS_ONLY"],
            remove_unused_columns=self.config["REMOVE_UNUSED_COLUMNS"],
            run_name=self.config["WANDB_RUN_NAME"],
            save_steps=self.config["SAVE_STEPS"],
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
                self.model = load_model(join(self.config["INPUT_DIR"], self.config["INPUT_FILE"]))
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
            eval_dataset=self.inputs["validation"]
        )
        trainer.train()

    def _predict(self):
        # trainer = Trainer(
        #     model=self.model,
        #     args=self.trainer_args
        # )
        #
        # raw_prediction = trainer.predict(test_dataset=self.inputs["test"])
        # predictions = []
        # print("predictions: " + str(len(raw_prediction.predictions)))
        # print(raw_prediction.predictions)
        # print("label_ids: " + str(len(raw_prediction.label_ids)))
        # print(raw_prediction.label_ids)
        # print("predictions printed, next is decoding")
        # for entry in raw_prediction.label_ids:
        #     output = self.tokenizer.batch_decode(entry, skip_special_tokens=True)
        #     predictions.append([x.replace('<pad>', '').replace('</s>', '').strip() for x in output])
        #
        # predictions = list(chain(*predictions))
        # print(predictions)
        print("start decoding")

        val_loader = DataLoader(dataset=self.inputs["test"], batch_size=8, num_workers=0)
        # Call validation function
        prediction, target = validation(self.tokenizer, self.model, val_loader)
        print("finished decoding")
        print("predictions:")
        print(prediction)
        print("target: ")
        print(target)


def validation(tokenizer, model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        model.to(device)
        for step, data in enumerate(loader, 0):
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            y_id = data['labels'].to(device)
            raw_prediction = model.generate(
                input_ids=ids,
                attention_mask=mask,
                num_beams=2,
                max_length=170,
                repetition_penalty=2.5,
                early_stopping=True,
                length_penalty=1.0
            )

    # Decode y_id and prediction #
    prediction = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=False) for p in raw_prediction]
    target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in y_id]

    predictions.extend(prediction)
    targets.extend(target)
    return predictions, targets
