import argparse

mode = 'simple'
if mode == 'simple':
    from models.simple.training_testing import *
    from models.simple.testing import *
else:
    from models.training_testing import *
    from models.testing import *


def main():
    log_dir = print_banner()
    if args.train_from_scratch:
        print(f'Training from scratch')
        model, model_params, criterion = train_model(log_dir)
        test_model(model, model_params, criterion, log_dir)
    if args.train_resume_checkpoint:
        print(f'Resume training from checkpoint {args.train_resume_checkpoint}')
        model, model_params, criterion = train_model(log_dir, train_resume_checkpoint=args.train_resume_checkpoint)
        test_model(model, model_params, criterion, log_dir)
    if args.test_saved_model_path:
        print(f'Loading saved model {args.test_saved_model_path} to test')
        check_point_dict = torch.load(args.test_saved_model_path)
        model = get_model(check_point_dict['model_params'])
        model.load_state_dict(check_point_dict['model_state_dict'])
        print(check_point_dict['criterion'])
        criterion = nn.BCEWithLogitsLoss()
        print(f"Log override to {check_point_dict['log_dir']}")
        test_model(model, check_point_dict['model_params'], criterion, check_point_dict['log_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and predict DeepFakes')

    parser.add_argument('--train_from_scratch', action='store_true', default=False, help='Train the model from scratch')
    parser.add_argument('--train_resume', dest='train_resume_checkpoint', default=False,
                        help='Resume the model training')
    parser.add_argument('--test_saved_model', dest='test_saved_model_path', default=False, help='Test the saved model')

    args = parser.parse_args()
    print(args)
    main()
