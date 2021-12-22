from datasets import load_dataset


def pre_process_web_nlg(path: str, name: str = None):
    dataset_union = load_dataset(path, name)
    container = DatasetContainer(dataset_union, ["train", "dev", "test"])
    return container


class DatasetContainer:

    def __init__(self, dataset, split_names):
        if split_names[0] in dataset:
            self.train_data = dataset[split_names[0]]
        else:
            self.train_data = None

        if split_names[1] in dataset:
            self.validation_data = dataset[split_names[1]]
        else:
            self.validation_data = None

        if split_names[2] in dataset:
            self.test_data = dataset[split_names[2]]
        else:
            self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_validation_data(self):
        return self.validation_data

    def get_test_data(self):
        return self.test_data
