from itertools import chain

import torch


def make_predictions(model, encoding, tokenizer):
    model_predictions = []
    model.eval()
    with torch.no_grad():
        input_ids = encoding['input_ids']
        attention_mask = encoding[i]['attention_mask']

        args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_length': 500,
            'bos_token_id': 0,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'use_cache': True
        }

        output = tokenizer.batch_decode(
            model.generate(**args),
            skip_special_tokens=True)
        model_predictions.append([x.replace('<pad>', '').replace('</s>', '').strip() for x in output])

        # flatten the predictions list which has the length of batch_size * number_of_batches
        model_predictions = list(chain(*model_predictions))
    return model_predictions
