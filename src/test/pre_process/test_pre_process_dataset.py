import unittest
from os.path import join
from datasets import load_dataset
from src.utils.config import load_config_from_yaml
from src.pre_process.preprocess import pre_process_dataset, DatasetContainer


class PreprocessDatasetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dataset = load_dataset("web_nlg", "release_v3.0_en")
        self.test_container = pre_process_dataset("web_nlg", "release_v3.0_en", ("train", "dev", "test"))
        self.path_to_yaml = join(".", "src", "test", "data", "test_experiment.yml")

    def test_pre_process_is_container(self):
        self.assertIsInstance(self.test_container, DatasetContainer)

    def test_pre_process_container_has_split_data(self):
        parameters = [
            {"actual": self.test_container.get_train_data, "expected": self.test_dataset["train"]},
            {"actual": self.test_container.get_validation_data, "expected": self.test_dataset["dev"]},
            {"actual": self.test_container.get_test_data, "expected": self.test_dataset["test"]}
        ]
        for parameter in parameters:
            with self.subTest(i=parameter):
                data = parameter["actual"]()
                self.assertEqual(data.data, parameter["expected"].data)

    def test_pre_process_from_yaml(self):
        container = self._get_test_container_from_yaml()
        self.assertIsInstance(container, DatasetContainer)

    def test_pre_process_from_yaml_container_has_split_data(self):
        container = self._get_test_container_from_yaml()

        parameters = [
            {"actual": container.get_train_data, "expected": self.test_dataset["train"]},
            {"actual": container.get_validation_data, "expected": self.test_dataset["dev"]},
            {"actual": container.get_test_data, "expected": self.test_dataset["test"]}
        ]

        for parameter in parameters:
            with self.subTest(i=parameter):
                data = parameter["actual"]()
                self.assertEqual(data.data, parameter["expected"].data)

    def _get_test_container_from_yaml(self):
        config_experiment = load_config_from_yaml(self.path_to_yaml)
        config_dataset = load_config_from_yaml(config_experiment["DATASET_CONFIG"])
        split_names = (
            config_dataset["NAME_TRAIN_DATASET"],
            config_dataset["NAME_VALIDATION_DATASET"],
            config_dataset["NAME_TEST_DATASET"]
        )
        container = pre_process_dataset(config_dataset["DATASET_PATH"], config_dataset["DATASET_NAME"], split_names)
        return container
