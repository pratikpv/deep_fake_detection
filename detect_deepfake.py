import argparse
from models.training_testing import *


def main():
    print_banner()
    train_model(args.train_method)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data pre-processing for DFDC')

    parser.add_argument('--train', dest='train_method', choices=['from_scratch', 'resume'],
                        help='Train the model')

    args = parser.parse_args()
    print(args)
    main()
