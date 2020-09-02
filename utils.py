import yaml
from datetime import datetime
import os
from pprint import pprint


def get_log_dir_name(create_logdir=True):
    current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
    config = load_config()
    log_dir = os.path.join(config['logging']['root_log_dir'], current_time_str)
    if create_logdir:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def load_config(config_file='config.yml'):
    with open(config_file, 'r') as c_file:
        config = yaml.safe_load(c_file)
    return config


def print_config(config=None):
    if config is None:
        config = load_config()
    pprint(config)


def print_line(print_len=None):
    if print_len is None:
        config = load_config()
    print('-' * config['printing']['line_len'])
