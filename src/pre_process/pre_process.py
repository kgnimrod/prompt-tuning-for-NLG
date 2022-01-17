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
            input_batch, max_length=3000, padding='max_length', return_tensors='pt')
        label_batch = tokenizer.batch_encode_plus(
            label_batch, max_length=3000, padding='max_length', return_tensors='pt')

        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        inputs.append(input_batch)
        labels.append(label_batch)
    return inputs, labels
