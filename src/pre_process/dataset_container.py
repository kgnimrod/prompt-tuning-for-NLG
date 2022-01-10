class DatasetContainer:

    def __init__(self, train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def get_train_data(self):
        return self.train_data

    def get_train_label(self):
        return self.train_labels

    def get_validation_data(self):
        return self.validation_data

    def get_validation_label(self):
        return self.validation_labels

    def get_test_data(self):
        return self.test_data

    def get_test_label(self):
        return self.test_labels
