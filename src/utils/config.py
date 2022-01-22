import yaml


def load_config_from_yaml(file):
    with open(file, 'r') as file:
        config = yaml.safe_load(file)
    return config
