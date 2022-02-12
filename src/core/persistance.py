from os import mkdir
from os.path import join, exists
import torch


def save_soft_prompt(soft_prompt, path, epochs, model_size, number_tokens):
    file = "soft_prompt_" + model_size + "_" + str(epochs) + "_epochs_" + str(number_tokens) + ".model"
    path = join(path, "soft_prompts")
    save_model(soft_prompt, path, file)


def save_model(model, path, filename="model.model"):
    path = validate_path(path)
    path = join(path, filename)
    torch.save(model, path)


def validate_path(path):
    if not exists(path):
        mkdir(path)
    return path


def load_model(path):
    return torch.load(path)


def save_checkpoint(epoch, model, optimizer, loss, path, filename="model"):
    path = validate_path(path)
    path = join(path, filename + ".model")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def save_state_dict(model, path, filename="model_state_dict"):
    path = validate_path(path)
    path = join(path, filename + ".model")
    torch.save(model.state_dict(), path)


def load_state_dict(model, path):
    model.load_state_dict(torch.load(path))
