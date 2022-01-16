import unittest
from os.path import join

import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, IntervalStrategy, Trainer

from src.utils.config import load_config_from_yaml


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.config = load_config_from_yaml(join(".", "src", "test", "data", "test_experiment.yml"))
        self.tokenizer = T5Tokenizer.from_pretrained(self.config["PRE_TRAINED_MODEL"])

    def test_webnlg_t5_base(self):

        config_dataset = load_config_from_yaml(self.config["DATASET_CONFIG"])
        self._train(config_dataset)

        # predict -> eval
        self.assertEqual(True, False)  # add assertion here

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

    def _train(self, config_dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = 4

        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=config["GPUS"])
        model.train()

        path = config_dataset["DATASET_PATH"]
        name = config_dataset["DATASET_NAME"]

        train_dataset = load_dataset(path, name, split=config_dataset["NAME_TRAIN_DATASET"])
        val_dataset = load_dataset(path, name, split=config_dataset["NAME_VALIDATION_DATASET"])
        if config_dataset["FLATTEN"]:
            train_dataset = train_dataset.flatten()
            val_dataset = val_dataset.flatten()

        train_dataset = train_dataset.rename_column(config_dataset["INPUT_IDS"], 'input_ids')
        train_dataset = train_dataset.rename_column(config_dataset["LABELS"], 'labels')

        val_dataset = val_dataset.rename_column(config_dataset["INPUT_IDS"], 'input_ids')
        val_dataset = val_dataset.rename_column(config_dataset["LABELS"], 'labels')

        train_dataset.shard(num_shards=4, index=0)

        train_dataset = train_dataset.map(self._tokenize, batched=True, batch_size=batch_size)
        val_dataset = val_dataset.map(self._tokenize, batched=True, batch_size=batch_size)

        for feature in train_dataset.features:
            if feature not in ['input_ids', 'attention_mask', 'labels']:
                train_dataset = train_dataset.remove_columns(feature)

        for feature in val_dataset.features:
            if feature not in ['input_ids', 'attention_mask', 'labels']:
                val_dataset = val_dataset.remove_columns(feature)

        train_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'])

        print("set training args")
        output_dir = "test-webnlg-t5-base"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
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

        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                          eval_dataset=val_dataset)

        print("start training")
        trainer.train()
        print("finished training")
        print("saving model")
        trainer.save_model(join("models", output_dir))


if __name__ == '__main__':
    unittest.main()
