import argparse
from models.training_testing import *
from models.testing import *


def main():
    print_banner()
    if args.train_method:
        model, model_params, criterion, log_dir = train_model(args.train_method)
    if args.test:
        test_model(model, model_params, criterion, log_dir)
    if args.test_saved_model_path:
        print(f'Loading model {args.test_saved_model_path}')
        check_point_dict = torch.load(args.test_saved_model_path)
        model = get_model(check_point_dict['model_params'])
        model.load_state_dict(check_point_dict['model_state_dict'])
        if check_point_dict['criterion'] == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception('Unknown criterion while testing saved model')
        print(f"Log override to {check_point_dict['log_dir']}")
        test_model(model, check_point_dict['model_params'], criterion, check_point_dict['log_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data pre-processing for DFDC')

    parser.add_argument('--train', dest='train_method', choices=['from_scratch', 'resume'],
                        help='Train the model')
    parser.add_argument('--test', action='store_true', default=False, help='Test the model')
    parser.add_argument('--test_saved_model', dest='test_saved_model_path',
                        default=False, help='Test the saved model')

    args = parser.parse_args()
    print(args)
    main()
