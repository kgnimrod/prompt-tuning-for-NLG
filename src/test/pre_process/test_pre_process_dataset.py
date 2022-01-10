import unittest
from os.path import join
from datasets import load_dataset
from src.utils.config import load_config_from_yaml
from src.pre_process.preprocess import pre_process_dataset
from src.pre_process.dataset_container import DatasetContainer


class PreprocessDatasetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dataset = load_dataset("web_nlg", "release_v3.0_en")
        self.test_config = {
            "DATASET_PATH": "web_nlg",
            "DATASET_NAME": "release_v3.0_en",
            "NAME_TRAIN_DATASET": "train",
            "NAME_VALIDATION_DATASET": "dev",
            "NAME_TEST_DATASET": "test"
        }
        self.test_container = pre_process_dataset(self.test_config)
        self.path_to_yaml = join(".", "src", "test", "data", "test_experiment.yml")

    def test_pre_process_is_container(self):
        self.assertIsInstance(self.test_container, DatasetContainer)

    def test_pre_process_from_yaml(self):
        config_experiment = load_config_from_yaml(self.path_to_yaml)
        config_dataset = load_config_from_yaml(config_experiment["DATASET_CONFIG"])
        container = pre_process_dataset(config_dataset)
        self.assertIsInstance(container, DatasetContainer)
