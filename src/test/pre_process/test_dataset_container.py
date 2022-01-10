import unittest
from datasets import load_dataset


class DatasetContainerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_train_data = load_dataset("web_nlg", "release_v3.0_en", "train")
        self.test_val_data = load_dataset("web_nlg", "release_v3.0_en", "dev")
        self.test_test_data = load_dataset("web_nlg", "release_v3.0_en", "test")

    def test_init_with_none_values(self):
        pass
