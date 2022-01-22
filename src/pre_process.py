from datasets import load_dataset


def create_list_of_batches(batch_size, num_batches, data, tokenizer, device):
    # Create List of batches for inputs and labels
    inputs = []
    labels = []
    for i in range(num_batches):
        input_batch = []
        label_batch = []
        for index, row in data[i*batch_size:i*batch_size+batch_size].iterrows():
            #          input_batch.append('translate from Graph to Text: '+row['input_text']+'</s>')
            #          label_batch.append(row['target_text']+'</s>')

            input_batch.append('translate from Graph to Text: '+row['input_text'])
            label_batch.append(row['target_text'])

        input_batch = tokenizer.batch_encode_plus(
            input_batch, padding=True, return_tensors='pt')
        label_batch = tokenizer.batch_encode_plus(
            label_batch, padding=True, return_tensors='pt')

        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

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

    return {'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}
