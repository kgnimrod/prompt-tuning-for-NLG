from itertools import chain

import torch


def make_predictions(model, encoding, tokenizer):
    model_predictions = []
    model.eval()
    with torch.no_grad():
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_length': 500,
            'bos_token_id': 0,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'use_cache': True
        }

        print(model.get_device())
        print(args['input_ids'].get_device())
        print(args['attention_mask'].get_device())
        output = tokenizer.batch_decode(
            model.generate(**args),
            skip_special_tokens=True)
        model_predictions.append([x.replace('<pad>', '').replace('</s>', '').strip() for x in output])

        # flatten the predictions list which has the length of batch_size * number_of_batches
        model_predictions = list(chain(*model_predictions))
    return model_predictions


def make_prediction_talk(model, tokenizer):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode('Translate from Graph to Text: Nie_Haisheng | birthDate | 1964-10-13', return_tensors='pt')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)
    print("Output:\n" + 500 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

    # encode context the generation is conditioned on
    input_ids = tokenizer.encode('Translate from Graph to Text: Nie_Haisheng | nationality | People\'s_Republic_of_China', return_tensors='pt')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)
    print("Output:\n" + 500 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

    # encode context the generation is conditioned on
    input_ids = tokenizer.encode('Translate from Graph to Text: Nie_Haisheng | mission | Shenzhou_10', return_tensors='pt')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)
    print("Output:\n" + 500 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
