from os.path import join
import pandas as pd
from datasets import load_dataset, Dataset


def create_list_of_batches(batch_size, num_batches, data, tokenizer):
    # Create List of batches for inputs and labels
    inputs = []
    labels = []
    for i in range(num_batches):
        input_batch = []
        label_batch = []
        for index, row in data[i*batch_size:i*batch_size+batch_size].iterrows():
            #          input_batch.append('translate from Graph to Text: '+row['input_text']+'</s>')
            #          label_batch.append(row['target_text']+'</s>')

            input_batch.append('translate from Graph to Text: '+row['input_ids'])
            label_batch.append(row['labels'])

        input_batch = tokenizer.batch_encode_plus(
            input_batch, padding=True, return_tensors='pt')
        label_batch = tokenizer.batch_encode_plus(
            label_batch, padding=True, return_tensors='pt')

        # input_batch = input_batch.to(device)
        # label_batch = label_batch.to(device)

        inputs.append(input_batch)
        labels.append(label_batch)
    return inputs, labels


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


def pre_process_amr_dataset(config):
    with open(join(config['DATASET_PATH'], config['DATASET_NAME'])) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    meaning_representations_not_flattened = list(
        filter(None, [line if not line.__contains__("#") else [] for line in lines])
    )
    target_sentences = list(filter(None, [line[8:] if line.__contains__("# ::snt") else [] for line in lines]))
    meaning_representations = []

    for i in range(len(meaning_representations_not_flattened)):
        if meaning_representations_not_flattened[i][0] == "(":
            j = i+1
            while meaning_representations_not_flattened[j][0] != "(":
                j += 1
                if j == len(meaning_representations_not_flattened):
                    break
            meaning_representations.append(
                ''.join(map(str, meaning_representations_not_flattened[i:j])).replace(' ', '')
            )

    # As for Web NLG and E2E the train/test split is roughly 90/10, so we also use this split for AMR
    train_dataset = Dataset.from_pandas(
        pd.DataFrame(
            list(zip(meaning_representations[:1164], target_sentences[:1164])), columns=['input_ids', 'labels']
        )
    )
    val_dataset = Dataset.from_pandas(
        pd.DataFrame(
            list(zip(meaning_representations[1164:1404], target_sentences[1164:1404])), columns=['input_ids', 'labels']
        )
    )
    test_dataset = Dataset.from_pandas(
        pd.DataFrame(
            list(zip(meaning_representations[1404:], target_sentences[1404:])), columns=['input_ids', 'labels']
        )
    )
    return {'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}


def pre_process_web_nlg_dataset(config):
    train_data = pd.read_csv(join(config['DATASET_PATH'], config['DATASET_NAME_TRAIN']))
    test_data = pd.read_csv(join(config['DATASET_PATH'], config['DATASET_NAME_TEST']))
    train_data = train_data.iloc[:len(train_data)-3, :]
    test_data = test_data.iloc[:len(test_data)-7, :]

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
