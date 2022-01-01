import unittest
from os.path import join

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.utils.config import load_config_from_yaml
from src.pre_process.preprocess import pre_process_dataset, serialize_dataset


class SimpleWebNlgExperimentTestCase(unittest.TestCase):
    def test_web_nlg_base(self):

        config = load_config_from_yaml(join(".", "config", "web_nlg.yml"))
        container = pre_process_dataset(
            config["DATASET_PATH"],
            config["DATASET_NAME"],
            (
                config["NAME_TRAIN_DATASET"],
                config["NAME_VALIDATION_DATASET"],
                config["NAME_TEST_DATASET"]
            )
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=config["GPUS"])

        serialized_data = serialize_dataset(container.get_test_data())

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
