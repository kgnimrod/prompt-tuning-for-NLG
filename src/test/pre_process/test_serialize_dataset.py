import unittest
from src.test.data.TestContainer import TestDatasetWebNlg
from src.pre_process.preprocess import serialize_dataset


class SerializeDatasetTestCase(unittest.TestCase):
    def test_serialize_labels(self):
        test_container = TestDatasetWebNlg()
        parameters = [test_container.container.get_train_data, test_container.container.get_test_data]

        for parameter in parameters:
            with self.subTest(i=parameter):
                expected = parameter().flatten()
                actual = serialize_dataset(parameter())
                print(actual)
                self.assertEquals(actual["lex.text"], expected["lex.text"])

    def test_serialize_inputs(self):
        test_container = TestDatasetWebNlg()
        parameters = [test_container.container.get_test_data]

        for parameter in parameters:
            with self.subTest(i=parameter):
                expected = parameter().flatten()
                actual = serialize_dataset(parameter())
                self.assertEqual(actual["original_triple_sets.otriple_set"], expected["original_triple_sets.otriple_set"])


if __name__ == '__main__':
    unittest.main()
