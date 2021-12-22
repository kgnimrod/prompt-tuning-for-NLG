import unittest
from datasets import load_dataset
from src.pre_processing.preprocess import DatasetContainer


class DatasetContainerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dataset = load_dataset("web_nlg", "release_v3.0_en")

    def test_init_with_none_values(self):
        parameters = [
            [[None, "dev", "test"], [None, self.test_dataset["dev"], self.test_dataset["test"]]]
            , [["train", None, "eval"], [self.test_dataset["train"], None, None]]
        ]

        for parameter in parameters:
            with self.subTest(i=parameter):
                container = DatasetContainer(self.test_dataset, parameter[0])
                actual = [container.get_train_data(), container.get_validation_data(), container.get_test_data()]
                self.assertEqual(actual, parameter[1])
