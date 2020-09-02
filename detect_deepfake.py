import torch
import sys
from utils import *


def print_banner():
    print_line()
    if torch.cuda.is_available():
        print('Using GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('No cuda-based GPU detected. Exiting')
        sys.exit(-1)

    log_dir = get_log_dir_name()
    print(f'LOG_DIR = {log_dir}')
    print_line()


def main():
    print_banner()


if __name__ == '__main__':
    main()
