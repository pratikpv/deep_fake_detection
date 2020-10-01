import torch
from utils import *
import os


def save_checkpoint(epoch=None, model=None, optimizer=None):
    model_class_name = type(model).__name__
    checkpoint_root_path = get_checkpoint_root_path()
    os.makedirs(checkpoint_root_path, exist_ok=True)
    check_point_path = os.path.join(checkpoint_root_path, model_class_name + '.chkpt')

    check_point_dict = {
        'epoch': epoch,
        'model_class_name': model_class_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(check_point_dict, check_point_path)


def load_checkpoint(model=None, optimizer=None):
    model_class_name = type(model).__name__
    check_point_path = os.path.join(get_checkpoint_root_path(), model_class_name + '.chkpt')
    check_point_dict = torch.load(check_point_path)

    if check_point_dict['model_class_name'] != model_class_name:
        raise Exception('Invalid checkpoint loading requested')
    model.load_state_dict(check_point_dict['model_state_dict'])
    optimizer.load_state_dict(check_point_dict['optimizer_state_dict'])
    return check_point_dict['epoch'], model, optimizer
