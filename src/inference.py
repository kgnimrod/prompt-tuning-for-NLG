from itertools import chain

import torch


def make_predictions(model, encoding, tokenizer):
    model_predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(encoding)):
            print('make_predictions: encoding: ', i)
            input_ids = encoding[i].input_ids
            attention_mask = encoding[i].attention_mask
            output = tokenizer.batch_decode(
                model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_length=400,
                    top_p=0.92,
                    top_k=0,
                    decoder_input_ids=input_ids,
                    attention_mask=attention_mask
                ),
                skip_special_tokens=True)
            model_predictions.append([x.replace('<pad>', '').replace('</s>', '').strip() for x in output])

        # flatten the predictions list which has the length of batch_size * number_of_batches
        model_predictions = list(chain(*model_predictions))
    return model_predictions
