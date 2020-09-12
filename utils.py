import torch
import sys
from utils import *
import cv2
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


def get_data_aug_plan_pkl_filename():
    config = load_config()
    return os.path.join(config['assets'], config['data_augmentation']['plan_pkl_file'])


def get_data_aug_plan_txt_filename():
    config = load_config()
    return os.path.join(config['assets'], config['data_augmentation']['plan_txt_file'])

def get_aug_metadata_folder():
    config = load_config()
    return os.path.join(config['assets'], config['data_augmentation']['metadata_folder'])

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


def print_banner():
    print_line()

    log_dir = get_log_dir_name()
    print(f'LOG_DIR = {log_dir}')
    print(f'PyTorch version = {torch.__version__}')
    if torch.cuda.is_available():
        print(f'PyTorch GPU = {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('PyTorch No cuda-based GPU detected.')
    print(f'OpenCV version  = {cv2.__version__}')

    print_line()
