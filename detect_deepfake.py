import argparse
from models.training_testing import *
from models.testing import *

def main():
    print_banner()
    if args.train_method:
        model, model_params, criterion, log_dir = train_model(args.train_method)
    if args.test:
        test_model(model, model_params, criterion, log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data pre-processing for DFDC')

    parser.add_argument('--train', dest='train_method', choices=['from_scratch', 'resume'],
                        help='Train the model')

    parser.add_argument('--test', action='store_true', default=False, help='Test the model')

    args = parser.parse_args()
    print(args)
    main()
