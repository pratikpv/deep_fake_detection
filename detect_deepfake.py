import argparse
from models.training_testing import *




def main():
    print_banner()
    train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data pre-processing for DFDC')

    parser.add_argument('--apply_aug_to_sample', action='store_true',
                        help='Apply augmentation and distractions to a file',
                        default=False)

    main()

