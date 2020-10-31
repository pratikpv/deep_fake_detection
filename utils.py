import torch
import sys
import cv2
import yaml
from datetime import datetime
import os
from pprint import pprint
import shutil


def load_config(config_file='config.yml'):
    with open(config_file, 'r') as c_file:
        config = yaml.safe_load(c_file)
    return config


def print_config(config=None):
    if config is None:
        config = load_config()
    pprint(config)


def copy_config(config_file='config.yml', dest=None):
    shutil.copy(config_file, dest)


def get_assets_path():
    config = load_config()
    return config['assets']


def get_train_data_path():
    config = load_config()
    return config['data_path']['train']


def get_validation_data_path():
    config = load_config()
    return config['data_path']['valid']


def get_test_data_path():
    config = load_config()
    return config['data_path']['test']


def get_backup_train_data_path():
    config = load_config()
    return config['data_path']['train_backup']


def get_train_frame_label_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['train_frame_label'])


def get_valid_frame_label_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['valid_frame_label'])


def get_test_frame_label_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['test_frame_label'])


def get_train_labels_csv_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['train_labels_csv_filename'])


def get_valid_labels_csv_filepath():
    config = load_config()
    return os.path.join(get_validation_data_path(), config['data_path']['valid_labels_csv_filename'])


def get_test_labels_csv_filepath():
    config = load_config()
    return os.path.join(get_test_data_path(), config['data_path']['test_labels_csv_filename'])


def get_train_optframe_label_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['train_optframe_label'])


def get_valid_optframe_label_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['valid_optframe_label'])


def get_test_optframe_label_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['test_optframe_label'])


def get_processed_train_data_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['processed_train_filename'])


def get_processed_validation_data_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['processed_valid_filename'])


def get_processed_test_data_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['processed_test_filename'])


def get_train_facecount_csv_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['train_faces_count'])


def get_valid_facecount_csv_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['valid_faces_count'])


def get_test_facecount_csv_filepath():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_path']['test_faces_count'])


def get_data_aug_plan_pkl_filename():
    config = load_config()
    return os.path.join(config['assets'], config['data_augmentation']['plan_pkl_filename'])


def get_data_aug_plan_txt_filename():
    config = load_config()
    return os.path.join(config['assets'], config['data_augmentation']['plan_txt_filename'])


def get_aug_metadata_path():
    config = load_config()
    return os.path.join(config['assets'], config['data_augmentation']['metadata'])


def get_compression_csv_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_augmentation']['compression_csv_filename'])


def get_video_integrity_data_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['data_augmentation']['integrity_csv_filename'])


def get_train_json_faces_data_path():
    config = load_config()
    return config['features']['train_json_faces']


def get_valid_json_faces_data_path():
    config = load_config()
    return config['features']['valid_json_faces']


def get_test_json_faces_data_path():
    config = load_config()
    return config['features']['test_json_faces']


def get_train_crop_faces_data_path():
    config = load_config()
    return config['features']['train_crop_faces']


def get_valid_crop_faces_data_path():
    config = load_config()
    return config['features']['valid_crop_faces']


def get_test_crop_faces_data_path():
    config = load_config()
    return config['features']['test_crop_faces']


def get_train_optical_flow_data_path():
    config = load_config()
    return config['features']['train_optical_flow']


def get_valid_optical_flow_data_path():
    config = load_config()
    return config['features']['valid_optical_flow']


def get_test_optical_flow_data_path():
    config = load_config()
    return config['features']['test_optical_flow']


def get_train_optical_png_data_path():
    config = load_config()
    return config['features']['train_optical_png']


def get_valid_optical_png_data_path():
    config = load_config()
    return config['features']['valid_optical_png']


def get_test_optical_png_data_path():
    config = load_config()
    return config['features']['test_optical_png']


def get_train_faces_cnn_features_data_path():
    config = load_config()
    return os.path.join(config['features']['train_faces_cnn'], get_default_cnn_encoder_name())


def get_valid_faces_cnn_features_data_path():
    config = load_config()
    return os.path.join(config['features']['valid_faces_cnn'], get_default_cnn_encoder_name())


def get_test_faces_cnn_features_data_path():
    config = load_config()
    return os.path.join(config['features']['test_faces_cnn'], get_default_cnn_encoder_name())


def get_faces_loc_video_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['features']['face_location_video_data_path'])


def get_default_cnn_encoder_name():
    config = load_config()
    return config['cnn_encoder']['default']


def get_log_dir_name(create_logdir=True):
    current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
    config = load_config()
    log_dir = os.path.join(config['logging']['root_log_dir'], current_time_str)
    if create_logdir:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_training_sample_size():
    config = load_config()
    return float(config['training']['train_size'])


def get_valid_sample_size():
    config = load_config()
    return float(config['training']['valid_size'])


def get_test_sample_size():
    config = load_config()
    return float(config['training']['test_size'])


def get_checkpoint_root_path():
    config = load_config()
    return os.path.join(get_assets_path(), config['training']['checkpoint_path'])


def get_training_params():
    config = load_config()
    return config['training']['params']


def get_log_params():
    config = load_config()
    return config['logging']


def print_line(print_len=None):
    if print_len is None:
        config = load_config()
    print('-' * config['logging']['line_len'])


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

    return log_dir


def create_assets_placeholder():
    os.makedirs(get_assets_path(), exist_ok=True)
    os.makedirs(get_aug_metadata_path(), exist_ok=True)


def alpha_sort_keys(item):
    return int(os.path.splitext(os.path.basename(item))[0])
