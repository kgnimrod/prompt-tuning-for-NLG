from os.path import join
from src.utils.config import load_config_from_yaml
from src.pre_process.preprocess import pre_process_dataset


class TestDatasetWebNlg:
    def __init__(self):
        self.path_to_config = join(".", "src", "test", "data", "test_config.yml")
        self.config = load_config_from_yaml(self.path_to_config)
        self.container = pre_process_dataset(self.config)
