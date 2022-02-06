import unittest
from os.path import join

import torch
from src.core.pre_process import pre_process_huggingface_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, IntervalStrategy, Trainer

from src.core.config import load_config_from_yaml


class SimpleHuggingfaceTrainerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config = load_config_from_yaml(join(".", "src", "test", "data", "test_experiment.yml"))
        self.tokenizer = T5Tokenizer.from_pretrained(self.config["PRE_TRAINED_MODEL"])
        self.model = T5ForConditionalGeneration.from_pretrained(self.config["PRE_TRAINED_MODEL"])
        self.datasets = {}
        self.parameters = {}

    def test_webnlg_t5_base(self):

        config_dataset = load_config_from_yaml(self.config["DATASET_CONFIG"])
        datasets = pre_process_huggingface_dataset(config_dataset)
        self.parameters["batch_size"] = 11
        self._prepare_datasets(datasets)
        self._to_device()
        self._train()

        # predict -> eval
        self.assertEqual(True, False)  # add assertion here

    def _prepare_datasets(self, datasets):
        train_dataset = datasets['train']
        train_dataset.shard(num_shards=4, index=0)
        train_dataset = train_dataset.map(self._tokenize, batched=True, batch_size=self.parameters["batch_size"])

        val_dataset = datasets['validation']
        val_dataset = val_dataset.map(self._tokenize, batched=True, batch_size=self.parameters["batch_size"])

        for feature in train_dataset.features:
            if feature not in ['input_ids', 'attention_mask', 'labels']:
                train_dataset = train_dataset.remove_columns(feature)

        for feature in val_dataset.features:
            if feature not in ['input_ids', 'attention_mask', 'labels']:
                val_dataset = val_dataset.remove_columns(feature)

        train_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'])

        self.datasets["train"] = train_dataset
        self.datasets["validation"] = val_dataset

    def _train(self):

        print("set training args")
        output_dir = self.config["OUTPUT_DIR"]
        training_args = TrainingArguments(
            output_dir=join("logs", output_dir),
            num_train_epochs=10,
            per_device_train_batch_size=self.parameters["batch_size"],
            per_device_eval_batch_size=self.parameters["batch_size"],
            eval_accumulation_steps=1,  # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
            prediction_loss_only=True,  # If I need to compute only loss and not other metrics-> will use less RAM
            learning_rate=0.001,
            evaluation_strategy=IntervalStrategy.STEPS,  # Run evaluation every eval_steps
            save_steps=1000,  # How often to save a checkpoint
            save_total_limit=1,  # Number of maximum checkpoints to save
            remove_unused_columns=True,  # Removes useless columns from the dataset
            run_name='run_name',  # Wandb run name
            logging_steps=1000,  # How often to log loss to wandb
            eval_steps=1000,  # How often to run evaluation on the val_set
            logging_first_step=False,  # Whether to log also the very first training step to wandb
            load_best_model_at_end=True,  # Whether to load the best model found at each evaluation.
            metric_for_best_model="loss",  # Use loss to evaluate best model.
            greater_is_better=False  # Best model is the one with the lowest loss, not highest.
        )

        self.model.train()
        trainer = Trainer(model=self.model, args=training_args, train_dataset=self.datasets["train"],
                          eval_dataset=self.datasets["validation"])

        print("start training")
        trainer.train()
        print("finished training")
        print("saving model")
        trainer.save_model(join("models", output_dir))

    def _tokenize(self, batch):
        inputs = []
        for item in batch['input_ids']:
            inputs.append(item[0][0])

        tokenized_input = self.tokenizer.batch_encode_plus(
            inputs, padding='max_length', max_length=500
        )

        labels = []
        for item in batch['labels']:
            labels.append(item[0])

        tokenized_labels = self.tokenizer.batch_encode_plus(
            labels, padding='max_length', max_length=500
        )
        tokenized_input['labels'] = tokenized_labels["input_ids"]

        return tokenized_input

    def _to_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # self.model = torch.nn.DataParallel(model, device_ids=config["GPUS"])
        for item in self.datasets:
            self.datasets[item]["input_ids"].to(device)
            self.datasets[item]["attention_mask"].to(device)
            self.datasets[item]["labels"].to(device)


if __name__ == '__main__':
    unittest.main()
