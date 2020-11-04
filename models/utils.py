import torch
import sys
from utils import *
from data_utils.utils import *
from quantification.utils import *
from models.DeepFakeDetectModel_1 import *
from models.DeepFakeDetectModel_2 import *
from models.DeepFakeDetectModel_3 import *
from models.DeepFakeDetectModel_4 import *
from models.DeepFakeDetectModel_5 import *
from models.DeepFakeDetectModel_6 import *


def print_batch_item(index, item, all_frames=False, simple=True):
    if simple:
        print_line()
        if item is None:
            print(f'{index} | None')
            return
        else:
            v_ids, frames, labels = item
            print(f'{index} | {v_ids} |frames={len(frames)}, shape={frames[0].shape} | {labels}')
            print_line()
            print(frames[0])
            print_line()

        print_line()


    else:
        print_line()
        print(f'index={index}')
        if item is None:
            print('None data')
            return
        v_ids, frames, labels = item
        print_line()
        print(f'v_ids={v_ids}')
        print_line()
        if all_frames:
            print(f'frames len = {len(frames)}')
            for f in frames:
                print(f'\t{f.shape}')
        else:
            print(f'frames len = {len(frames)}, shape = {frames[0].shape}')
        print_line()
        print(f'labels = {labels}')
        print_line()


def global_minibatch_number(epoch, batch_id, batch_size):
    """
    get global counter of iteration for smooth plotting
    @param epoch: epoch
    @param batch_id: the batch number
    @param batch_size: batch size
    @return: global counter of iteration
    """
    return epoch * batch_size + batch_id


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    """
    for i, b in enumerate(batch):
        print_batch_item(i, b)
    """
    batch = tuple(zip(*batch))
    return batch


def get_predictions(output):
    # return torch.argmax(output, dim=1)
    return torch.round(torch.sigmoid(output))


def get_probability(output):
    return torch.sigmoid(output)


def get_model(model_params):
    model = None
    # validate model_params
    model_batch_mapping = {
        'stacked': ['DeepFakeDetectModel_2', 'DeepFakeDetectModel_3', 'DeepFakeDetectModel_4',
                    'DeepFakeDetectModel_5'],
        'simple': ['DeepFakeDetectModel_6']
    }

    if model_params['model_name'] not in model_batch_mapping[model_params['batch_format']]:
        raise Exception("model_name and batch_format does not match")

    if model_params['model_name'] == 'DeepFakeDetectModel_2':
        model = DeepFakeDetectModel_2(frame_dim=model_params['imsize'], max_num_frames=model_params['max_num_frames'],
                                      encoder_name=model_params['encoder_name'],
                                      lstm_hidden_dim=model_params['lstm']['hidden_dim'],
                                      lstm_num_layers=model_params['lstm']['num_layers'],
                                      lstm_dropout=model_params['lstm']['dropout'],
                                      )
    elif model_params['model_name'] == 'DeepFakeDetectModel_3':
        model = DeepFakeDetectModel_3(frame_dim=model_params['imsize'])
    elif model_params['model_name'] == 'DeepFakeDetectModel_4':
        model = DeepFakeDetectModel_4(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    elif model_params['model_name'] == 'DeepFakeDetectModel_5':
        model = DeepFakeDetectModel_5(frame_dim=model_params['imsize'], max_num_frames=model_params['max_num_frames'],
                                      encoder_name=model_params['encoder_name'])
    elif model_params['model_name'] == 'DeepFakeDetectModel_6':
        model = DeepFakeDetectModel_6(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    else:
        raise Exception("Unknown model name passed")

    return model
