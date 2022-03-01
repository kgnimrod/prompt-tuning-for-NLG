import numpy as np
import torch
from datasets import load_metric


def compute_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("bleurt")
    metric4 = load_metric("bertscore")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    bleurt = metric3.compute(predictions=predictions, references=labels)["bleurt"]
    bertscore = metric4.compute(predictions=predictions, references=labels)["bertscore"]

    return {
        "precision": precision,
        "recall": recall,
        "bleurt": bleurt,
        "bertscore": bertscore
    }


def validation(tokenizer, model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        model.to(device)
        for step, data in enumerate(loader, 0):
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            y_id = data['labels'].to(device)
            raw_prediction = model.generate(
                input_ids=ids,
                attention_mask=mask,
                num_beams=2,
                max_length=170,
                repetition_penalty=2.5,
                early_stopping=True,
                length_penalty=1.0
            )

            # Decode y_id and prediction #
            prediction = [
                tokenizer.decode(
                    p, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ) for p in raw_prediction
            ]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in y_id]

            predictions.extend(prediction)
            targets.extend(target)
    return predictions, targets
