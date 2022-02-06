from os.path import join
import argparse

from src.core.experiment import Experiment
from src.core.config import load_config_from_yaml


def __main__():
    parser = argparse.ArgumentParser(add_help=False)
    args = parse_arguments(parser)
    file = join("experiments", args.experiment)
    config = load_config_from_yaml(file)
    experiment = Experiment(config)
    experiment.run()


def parse_arguments(parser):
    parser.add_argument('--experiment', default="undefined",
                        help='Name of experiment to run')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __main__()
