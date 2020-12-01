import argparse
from models.meta.MRI_GAN.training import *
from utils import *


def main():
    log_dir = print_banner()
    if args.train_from_scratch:
        print(f'Training MRI-GAN from scratch')
        train_MRI_GAN_model(log_dir=log_dir)
    if args.train_resume_checkpoint_dir:
        print(f'Resume pix2pix training from checkpoint {args.train_resume_checkpoint_dir}')
        train_MRI_GAN_model(log_dir, train_resume_dir=args.train_resume_checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pix2Pix GAN')

    parser.add_argument('--train_from_scratch', action='store_true',
                        help='Train Pix2Pix GAN from scratch',
                        default=False)
    parser.add_argument('--train_resume', dest='train_resume_checkpoint_dir', default=False,
                        help='Resume training Pix2Pix GAN')
    args = parser.parse_args()
    main()
