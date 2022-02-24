from os.path import join
import pandas as pd
from datasets import load_dataset, Dataset


def pre_process_huggingface_dataset(config):
    path = config["DATASET_PATH"]
    name = config["DATASET_NAME"]

    train_dataset = load_dataset(path, name, split=config["NAME_TRAIN_DATASET"])
    val_dataset = load_dataset(path, name, split=config["NAME_VALIDATION_DATASET"])
    test_dataset = load_dataset(path, name, split=config["NAME_TEST_DATASET"])
    if config["FLATTEN"]:
        train_dataset = train_dataset.flatten()
        val_dataset = val_dataset.flatten()
        test_dataset = test_dataset.flatten()

    train_dataset = train_dataset.rename_column(config["INPUT_IDS"], 'input_ids')
    train_dataset = train_dataset.rename_column(config["LABELS"], 'labels')

    val_dataset = val_dataset.rename_column(config["INPUT_IDS"], 'input_ids')
    val_dataset = val_dataset.rename_column(config["LABELS"], 'labels')

    test_dataset = test_dataset.rename_column(config["INPUT_IDS"], 'input_ids')
    test_dataset = test_dataset.rename_column(config["LABELS"], 'labels')

    columns_to_delete = []
    for feature in train_dataset.features:
        if feature != "input_ids" and feature != "labels":
            columns_to_delete.append(feature)

    train_dataset = train_dataset.remove_columns(columns_to_delete)
    val_dataset = val_dataset.remove_columns(columns_to_delete)
    test_dataset = test_dataset.remove_columns(columns_to_delete)

    return {'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}


def pre_process_custom_dataset(config):
    train_data = pd.read_csv(join(config['DATASET_PATH'], config['DATASET_NAME_TRAIN']))
    test_data = pd.read_csv(join(config['DATASET_PATH'], config['DATASET_NAME_TEST']))
    train_data = train_data.iloc[:len(train_data)-config['TRIM_TRAIN'], :]
    test_data = test_data.iloc[:len(test_data)-config['TRIM_TEST'], :]

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data, columns=[config["INPUT_IDS"], config["LABELS"]]))
    train_dataset = train_dataset.rename_column(config["INPUT_IDS"], 'input_ids')
    train_dataset = train_dataset.rename_column(config["LABELS"], 'labels')
    split_dict = train_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dict['train']
    val_dataset = split_dict['test']

    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data, columns=[config["INPUT_IDS"], config["LABELS"]]))
    test_dataset = test_dataset.rename_column(config["INPUT_IDS"], 'input_ids')
    test_dataset = test_dataset.rename_column(config["LABELS"], 'labels')

    return {'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}


def sample(datasets, sample_size_train, sample_size_eval=None):
    # aligned_datasets = align_datasets(datasets)
    aligned_datasets = datasets
    if sample_size_eval is None:
        sample_size_eval = sample_size_train
    sampled_train = None
    sampled_val = None
    sampled_test = None
    for key in aligned_datasets:
        sample_percentage = sample_size_train * 100 / len(aligned_datasets[key]["train"])
        sample_size_val = int(sample_percentage * len(aligned_datasets[key]["validation"]) / 100)

        sampled_data = aligned_datasets[key]["train"].shuffle().select([i for i in range(0, sample_size_train)])
        sampled_train = _combine(sampled_train, sampled_data)

        sampled_val = _combine(
            sampled_val, aligned_datasets[key]["validation"].shuffle().select([i for i in range(0, sample_size_val)])
        )
        sampled_test = _combine(
            sampled_test, aligned_datasets[key]["test"].shuffle().select([i for i in range(0, sample_size_eval)])
        )

    sampled_train = sampled_train.shuffle()
    sampled_val = sampled_val.shuffle()
    sampled_test = sampled_test.shuffle()
    return {'train': sampled_train, 'validation': sampled_val, 'test': sampled_test}


def combine(datasets):
    sampled_train = None
    sampled_val = None
    sampled_test = None
    for key in datasets:
        sampled_train = _combine(sampled_train, datasets[key]["train"])
        sampled_val = _combine(sampled_val, datasets[key]["validation"])
        sampled_test = _combine(sampled_test, datasets[key]["test"])

    sampled_train = sampled_train.shuffle()
    sampled_val = sampled_val.shuffle()
    sampled_test = sampled_test.shuffle()
    return {'train': sampled_train, 'validation': sampled_val, 'test': sampled_test}


def _combine(dataset_target, dataset_source):
    if dataset_target is None:
        dataset_target = dataset_source
    else:
        pandas = dataset_target.to_pandas()
        pandas.append(dataset_source.to_pandas())
        dataset_target = Dataset.from_pandas(pandas)
    return dataset_target
