import torch
import sys
from utils import *
from data_utils.utils import *
from models.DeepFakeDetectModel_1 import *
from data_utils.datasets import DFDCDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import multiprocessing
import numpy as np
import torchvision
from features.encoders import *
from torchvision.transforms import transforms
import torch.nn.functional as F


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


def my_collate(batch):
    # print(f'entering my_collate')
    batch = list(filter(lambda x: x is not None, batch))
    # for i, b in enumerate(batch):
    #    print_batch_item(i, b)
    # v_ids, frames, labels = batch

    batch = tuple(zip(*batch))
    # batch = torch.utils.data.dataloader.default_collate(batch)
    return batch


def train_model():
    log_dir = get_log_dir_name()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")

    train_data = get_all_training_video_filepaths(get_train_data_path())
    valid_data = get_all_validation_video_filepaths(get_validation_data_path())
    # test_data = get_all_test_video_filepaths(get_test_data_path())

    train_data_len = len(train_data)
    valid_data_len = len(valid_data)
    # test_data_len = len(test_data)

    sample_size = get_training_sample_size()
    if sample_size > 0.0:
        train_data_len = int(train_data_len * sample_size)
        valid_data_len = int(valid_data_len * sample_size)
        # test_data_len = int(test_data_len * sample_size)
        train_data = random.sample(train_data, train_data_len)
        valid_data = random.sample(valid_data, valid_data_len)
        # test_data = random.sample(test_data, test_data_len)

    encoder_name = get_default_cnn_encoder_name()
    imsize = encoder_params[encoder_name]["imsize"]

    train_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor()
    ])

    test_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor()
    ])

    train_dataset = DFDCDataset(train_data, mode='train', transform=train_transform, frame_dim=imsize)
    valid_dataset = DFDCDataset(valid_data, mode='valid', transform=test_transform, frame_dim=imsize)

    batch_size = 32
    # num_workers = multiprocessing.cpu_count() - 2
    num_workers = 0
    epochs = 20
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=False, collate_fn=my_collate)
    # ,   collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))

    model = DeepFakeDetectModel_1(frame_dim=imsize).to(device)
    lr = 0.0001
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f'train_data_len {train_data_len}')

    tqdm_train_descr_format = "Training model: Epoch Accuracy = {:02.4f}%, Loss = {:.8f}"
    tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'))
    tqdm_train_obj = tqdm(range(epochs), desc=tqdm_train_descr)

    for e in tqdm_train_obj:
        model.train(True)
        model, epoch_accuracy, epoch_loss = train_epoch(epoch=e, model=model, criterion=criterion, optimizer=optimizer,
                                                        train_loader=train_loader, batch_size=batch_size, device=device,
                                                        log_dir=log_dir)

        tqdm_descr = tqdm_train_descr_format.format(epoch_accuracy, epoch_loss)
        tqdm_train_obj.set_description(tqdm_descr)
        tqdm_train_obj.update()


def train_epoch(epoch=None, model=None, criterion=None, optimizer=None, train_loader=None, batch_size=None, device=None,
                log_dir=None):
    losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0

    for idx, samples in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = len(samples[0])
        frames_ = samples[1]
        frames = torch.stack(frames_).to(device)
        # print(f'{idx} | frames:{frames.shape}')
        labels = torch.stack(samples[2]).type(torch.float).to(device)
        predictions = model(frames)
        predictions = torch.squeeze(predictions, dim=1)
        # predictions = torch.randn(batch_size, device=device)
        batch_loss = criterion(predictions, labels)
        # batch_loss = F.binary_cross_entropy_with_logits(predictions, labels)
        # predicted = torch.max(predictions.data, 1)[1]
        # print(f'fpredictions = {predictions}')
        # print(f'labels = {labels}')
        batch_corr = (predictions == labels).sum()
        total_samples += batch_size
        total_correct += batch_corr.item()
        losses.append(batch_loss.item())
        accuracies.append(batch_corr.item())

        batch_loss.backward()
        optimizer.step()
        # print(f'{idx} batch_corr:{batch_corr}, batch_loss:{batch_loss.item()}')

    epoch_accuracy = total_correct / total_samples
    mean_epoch_loss = np.mean(losses)

    return model, epoch_accuracy, mean_epoch_loss
