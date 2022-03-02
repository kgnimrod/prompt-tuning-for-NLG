import numpy as np
import torch
from datasets import load_metric


def predict(tokenizer, model, loader):
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
    return {"predictions": predictions, "targets": targets}


def compute_scores(predictions, targets):
    print(len(predictions))
    print(len(targets))

    rouge = _compute_rouge_l(predictions=predictions, targets=targets)
    # ter = compute_score(predictions=predictions, labels=targets, name="ter")
    bertscore = _compute_bertscore(predictions=predictions, targets=targets)
    bleurt = _compute_bleurt(predictions=predictions, targets=targets)
    meteor = compute_score(predictions=predictions, labels=targets, name="meteor")

    scores = {
        "bertscore": bertscore["scores"],
        "bleurt": bleurt["scores"],
        "rougeL": rouge["scores"]
    }

    means = {
        "bertscore_precision": bertscore["means"]["precision"],
        "bertscore_recall": bertscore["means"]["recall"],
        "bertscore_f1": bertscore["means"]["f1"],
        "bleurt_f1": bleurt["means"]["f1"],
        "meteor": meteor,
        # "ter": ter["score"],
        "rougeL_f1": rouge["means"]["f1"],
    }

    return {"scores": scores, "means": means}


def compute_score(predictions, labels, name, checkpoint=None, **kwargs):
    metric = load_metric(name, checkpoint)
    metric.add_batch(predictions=predictions, references=labels)
    score = metric.compute(**kwargs)
    return score


def _compute_bertscore(predictions, targets):
    scores = compute_score(predictions=predictions, labels=targets, name="bertscore", lang="en")
    mean = {
        "precision": np.mean(np.array(scores["precision"])),
        "recall": np.mean(np.array(scores["recall"])),
        "f1": np.mean(np.array(scores["f1"]))
    }
    return {"scores": scores, "means": mean}


def _compute_bleurt(predictions, targets):
    scores = compute_score(predictions=predictions, labels=targets, name="bleurt", checkpoint="bleurt-large-512")
    mean = {
        "f1": np.mean(np.array(scores["scores"]))
    }
    return {"scores": scores, "means": mean}


def _compute_rouge_l(predictions, targets):
    scores = compute_score(predictions=predictions, labels=targets, name="rouge")
    print(scores)
    mean = {
        "f1": np.mean(np.array(scores["rougeL"].mid.fmeasure))
    }
    return {"scores": scores["rougeL"], "means": mean}
