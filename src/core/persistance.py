from os.path import join
import torch


def save_soft_prompt(soft_prompt, path, challenge_name, epochs, model_size, number_tokens):
    file = join("soft_prompt_", challenge_name, "_", model_size, "_", epochs, "_epochs_", number_tokens, ".model")
    path = join(path, "soft_prompts", file)
    save_model(soft_prompt, path)


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)
