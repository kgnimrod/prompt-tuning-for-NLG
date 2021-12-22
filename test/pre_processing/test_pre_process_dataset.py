import unittest
from datasets import load_dataset
from src.pre_processing.preprocess import pre_process_web_nlg, DatasetContainer


class PreprocessDatasetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dataset = load_dataset("web_nlg", "release_v3.0_en")
        self.test_container = pre_process_web_nlg("web_nlg", "release_v3.0_en")

    def test_pre_process_web_nlg_is_container(self):
        self.assertIsInstance(self.test_container, DatasetContainer)

    def test_pre_process_wb_nlg_has_train_data(self):
        data = self.test_container.get_train_data()
        self.assertEqual(data.data, self.test_dataset["train"].data)

    def test_pre_process_wb_nlg_has_validation_data(self):
        data = self.test_container.get_validation_data()
        self.assertEqual(data.data, self.test_dataset["dev"].data)

    def test_pre_process_wb_nlg_has_test_data(self):
        data = self.test_container.get_test_data()
        self.assertEqual(data.data, self.test_dataset["test"].data)
