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

    return {'train': sampled_train, 'validation': sampled_val, 'test': sampled_test}


def combine(datasets):
    sampled_train = None
    sampled_val = None
    sampled_test = None
    for key in datasets:
        sampled_train = _combine(sampled_train, datasets[key]["train"])
        sampled_val = _combine(sampled_val, datasets[key]["validation"])
        sampled_test = _combine(sampled_test, datasets[key]["test"])

    return {'train': sampled_train, 'validation': sampled_val, 'test': sampled_test}


def _combine(dataset_target, dataset_source):
    if dataset_target is None:
        dataset_target = dataset_source
    else:
        pandas = dataset_target.to_pandas()
        pandas.append(dataset_source.to_pandas())
        dataset_target = Dataset.from_pandas(pandas)
    return dataset_target

# As for Web NLG and E2E the datasets are already available as csv. For Abstract Meaning Representation (AMR), the official web page only provides a text file
# ,so we process this file to extract the meaning representations and the target sentences and save the results as csv
def preprocess_amr():
    with open('data/amr/amr-bank-struct-v3.0.txt') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    meaning_representations_not_flattened = list(filter(None, [line if not line.__contains__("#") else [] for line in lines]))
    target_sentences = list(filter(None, [line[8:] if line.__contains__("# ::snt") else [] for line in lines]))
    meaning_representations = []

    for i in range(len(meaning_representations_not_flattened)):
        if meaning_representations_not_flattened[i][0] == "(":
            j = i+1
            while meaning_representations_not_flattened[j][0] != "(":
                j +=1
                if j == len(meaning_representations_not_flattened): break
            meaning_representations.append(''.join(map(str, meaning_representations_not_flattened[i:j])).replace(' ', ''))

    # As for Web NLG and E2E the train/test split is roughly 90/10, so we also use this split for AMR
    pd.DataFrame(list(zip(meaning_representations[:1404], target_sentences[:1404])), columns=['input_text','target_text']).to_csv('data/amr/abstract_meaning_representation_train.csv', index=False)
    pd.DataFrame(list(zip(meaning_representations[1404:], target_sentences[1404:])), columns=['input_text','target_text']).to_csv('data/amr/abstract_meaning_representation_test.csv', index=False)

# Prepend <H>, <R>, <T> (Head, Relation, Tail) Tokens before each triple element in the input column as a second fine tuning technique for Web NLG
def replace_original_triples_with_tokens(data):
  last_token = 'tail'
  for i in range(len(data)):
    split_list = data.iloc[i]['input_text'].split(' | ')
    for j in range(len(split_list)):
      if last_token == 'tail':
        split_list[j] = '<H> ' + split_list[j]
        last_token = 'head'
      elif last_token == 'head':
        split_list[j] = ' <R> ' + split_list[j]
        last_token = 'relation'
      else:
        if last_token == 'relation' and j == len(split_list)-1:
          split_list[j] = ' <T> ' + split_list[j]
          last_token = 'tail'
          data.iloc[i]['input_text'] = ''.join(split_list)
        else:
          split_list[j] = ' <T> ' + split_list[j][:split_list[j].index('&')+2] + ' <H>' + split_list[j][split_list[j].index('&')+2:]
          last_token = 'head'
  return data

# For each equal input text, create a list of all references and finally add them as a new column
def create_references_lists(data):

    # then find the triple which has the highest occurence count in the input_text columns
    # This will be needed to later equalize the lengths of the reference lists for score calculation
    max_occurrence = max(data.groupby(['input_text']).size())
    references_list = []
    inputs_grouped = data.groupby(['input_text'])

    for i in range(len(data)):
        references = list(inputs_grouped.get_group(data.iloc[i]['input_text'])['target_text'])
        for j in range(max_occurrence-len(references)): references.append('')
        references_list.append(references)
    return references_list
    # Now add the references lists as new column to the sorted dataframes
    #references_list_web_nlg = create_references_lists(test_data_web_nlg)
    #test_data_web_nlg['references_list'] = references_list_web_nlg
    #test_data_e2e['references_list'] = create_references_lists(test_data_e2e)
    #test_data_amr['references_list'] = create_references_lists(test_data_amr)

def create_references_files_for_evaluation(data, path):
    inputs_grouped = data.groupby(['input_text'])
    max_occurrence = max(data.groupby(['input_text']).size())
    for i in range(max_occurrence):
        with open(path + 'reference' + str(i), 'w') as file:
            for name, group in inputs_grouped:
                file.write((group.iloc[i][1] + '\n') if len(group) > i else '\n')

    #create_references_files_for_evaluation(test_data_web_nlg, 'data/web_nlg/test/')
    #create_references_files_for_evaluation(test_data_amr, 'data/amr/test/')
