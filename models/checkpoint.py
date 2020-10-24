import torch
from utils import *
import os


def save_checkpoint(epoch=None, model=None, model_params=None,
                    optimizer=None, criterion=None, log_dir=None):
    model_class_name = type(model).__name__
    checkpoint_root_path = log_dir
    os.makedirs(checkpoint_root_path, exist_ok=True)
    check_point_path = os.path.join(checkpoint_root_path, model_class_name + '.chkpt')

    check_point_dict = {
        'epoch': epoch,
        'model_class_name': model_class_name,
        'model_params': model_params,
        'criterion': criterion,
        'log_dir': log_dir,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(check_point_dict, check_point_path)


def load_checkpoint(model=None, optimizer=None, check_point_path=None):
    model_class_name = type(model).__name__
    check_point_dict = torch.load(check_point_path)

    if check_point_dict['model_class_name'] != model_class_name:
        raise Exception('Invalid checkpoint loading requested')
    model.load_state_dict(check_point_dict['model_state_dict'])
    optimizer.load_state_dict(check_point_dict['optimizer_state_dict'])
    return check_point_dict['epoch'], model, optimizer, check_point_dict['model_params'], check_point_dict['log_dir']
