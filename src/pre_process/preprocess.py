from datasets import load_dataset
import pandas as pd

from src.pre_process.dataset_container import DatasetContainer


def pre_process_dataset(config):
    train_dataset = load_dataset(config["DATASET_PATH"], config["DATASET_NAME"], split=config["NAME_TRAIN_DATASET"])
    val_dataset = load_dataset(config["DATASET_PATH"], config["DATASET_NAME"], split=config["NAME_VALIDATION_DATASET"])
    test_dataset = load_dataset(config["DATASET_PATH"], config["DATASET_NAME"], split=config["NAME_TEST_DATASET"])

    # train data
    train_data_df = pd.DataFrame(train_dataset["original_triple_sets"], columns=['otriple_set'])
    train_data_df.rename(columns={'otriple_set': 'input_ids'}, inplace=True)
    train_labels_df = pd.DataFrame(train_dataset["lex"], columns=['text'])
    train_labels_df.rename(columns={'text': 'labels'}, inplace=True)
    train_labels_df["labels"] = train_labels_df["labels"].apply(lambda x: list(map(int, x)))
    train_data, train_labels = _compute_data_and_labels(train_data_df, train_labels_df, 5)

    # validation data
    val_data_df = pd.DataFrame(val_dataset["original_triple_sets"], columns=['otriple_set'])
    val_data_df.rename(columns={'otriple_set': 'input_ids'}, inplace=True)
    val_labels_df = pd.DataFrame(val_dataset["lex"], columns=['text'])
    val_labels_df.rename(columns={'text': 'labels'}, inplace=True)
    val_data, val_labels = _compute_data_and_labels(val_data_df, val_labels_df, 5)

    # test data
    test_data_df = pd.DataFrame(test_dataset["original_triple_sets"], columns=['otriple_set'])
    test_data_df.rename(columns={'otriple_set': 'input_ids'}, inplace=True)
    test_labels_df = pd.DataFrame(test_dataset["lex"], columns=['text'])
    test_labels_df.rename(columns={'text': 'labels'}, inplace=True)
    test_data, test_labels = _compute_data_and_labels(test_data_df, test_labels_df, 5)

    container = DatasetContainer(train_data, train_labels, val_data, val_labels, test_data, test_labels)
    return container


def _compute_data_and_labels(data_df, labels_df, label_len=1):
    data = []
    for entry in data_df['input_ids']:
        entry = 'WebNLG: '+entry[0][0]+'</s>'
        data.append(entry)

    labels = []
    for entries in labels_df['labels']:
        label = []
        for entry in entries:
            entry = entry+'</s>'
            label.append(entry)

        labels.append(label)

    return data, labels
