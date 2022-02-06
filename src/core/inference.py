from itertools import chain

import torch


def make_predictions(model, encoding, tokenizer, use_embeddings=False):
    model_predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(encoding)):
            print('make_predictions: encoding: ', i)
            input_ids = encoding[i]['input_ids']
            attention_mask = encoding[i]['attention_mask']

            args = {
                'input_ids': input_ids,
                # 'decoder_input_ids': input_ids,
                'max_length': 400,
                'bos_token_id': 0,
                'pad_token_id': 0,
                'eos_token_id': 1,
                'use_cache': True,
                'attention_mask': attention_mask
            }
            if use_embeddings:
                args['inputs_embeds'] = model.embed_tokens(input_ids)
                args['decoder_input_ids'] = input_ids
                args['attention_mask'] = model.extend_attention_mask(attention_mask)
                del args['input_ids']

            output = tokenizer.batch_decode(
                model.generate(**args),
                skip_special_tokens=True)
            model_predictions.append([x.replace('<pad>', '').replace('</s>', '').strip() for x in output])

        # flatten the predictions list which has the length of batch_size * number_of_batches
        model_predictions = list(chain(*model_predictions))
    return model_predictions
