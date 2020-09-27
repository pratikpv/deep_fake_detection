import torch
import sys
from utils import *
from data_utils.utils import *
from models.DeepFakeDetectModel_1 import *
from models.DeepFakeDetectModel_2 import *
from data_utils.datasets import DFDCDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import multiprocessing
import numpy as np
import torchvision
from features.encoders import *
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


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

    # train_data = get_all_training_video_filepaths(get_train_data_path())
    train_data = get_processed_training_video_filepaths(get_train_data_path())
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
        sample_random = True
        if sample_random:
            train_data = random.sample(train_data, train_data_len)
            valid_data = random.sample(valid_data, valid_data_len)
            # test_data = random.sample(test_data, test_data_len)
        else:
            train_data = train_data[0: train_data_len]
            valid_data = valid_data[0: valid_data_len]
            # test_data = test_data[0:test_data_len]

    encoder_name = get_default_cnn_encoder_name()
    imsize = encoder_params[encoder_name]["imsize"]

    train_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor()
    ])

    valid_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor()
    ])

    max_num_frames = 20
    train_dataset = DFDCDataset(train_data, mode='train', transform=train_transform, max_num_frames=max_num_frames,
                                frame_dim=imsize)
    valid_dataset = DFDCDataset(valid_data, mode='valid', transform=valid_transform, max_num_frames=max_num_frames,
                                frame_dim=imsize)

    batch_size = 8
    num_workers = multiprocessing.cpu_count() - 2
    # num_workers = 16
    epochs = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=my_collate)
    print(f'train_loader.batch_size {train_loader.batch_size}')
    model = DeepFakeDetectModel_2(frame_dim=imsize, max_num_frames=max_num_frames, encoder_name=encoder_name).to(device)
    lr = 0.0001
    criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f'train_data_len {train_data_len}')

    tqdm_train_descr_format = "Training model: Mean train accuracy = {:02.4f}%, mean train Loss = {:.8f}"
    tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'))
    tqdm_train_obj = tqdm(range(epochs), desc=tqdm_train_descr)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))

    model_train_accuracies = []
    model_train_losses = []
    for e in tqdm_train_obj:
        model, t_epoch_accuracy, t_epoch_loss = train_epoch(epoch=e, model=model, criterion=criterion,
                                                            optimizer=optimizer,
                                                            data_loader=train_loader, batch_size=batch_size,
                                                            device=device,
                                                            log_dir=log_dir, sum_writer=train_writer)
        model_train_accuracies.append(t_epoch_accuracy)
        model_train_losses.append(t_epoch_loss)

        tqdm_descr = tqdm_train_descr_format.format(np.mean(model_train_accuracies), np.mean(model_train_losses))
        tqdm_train_obj.set_description(tqdm_descr)
        tqdm_train_obj.update()

        train_writer.add_scalar('Training: loss per epoch', t_epoch_loss, e)
        train_writer.add_scalar('Training: accuracy per epoch', t_epoch_accuracy, e)

        v_epoch_accuracy, v_epoch_loss = valid_epoch(epoch=e, model=model, criterion=criterion, optimizer=optimizer,
                                                     data_loader=valid_loader, batch_size=batch_size, device=device,
                                                     log_dir=log_dir, sum_writer=train_writer)

        train_writer.add_scalar('Validation: loss per epoch', v_epoch_loss, e)
        train_writer.add_scalar('Validation: accuracy per epoch', v_epoch_accuracy, e)


def train_epoch(epoch=None, model=None, criterion=None, optimizer=None, data_loader=None, batch_size=None, device=None,
                log_dir=None, sum_writer=None):
    losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0
    train_loader_len = len(data_loader)
    model.train(True)

    tqdm_b_descr_format = "Training. Batch# {:05} Accuracy = {:02.4f}%, Loss = {:.8f}"
    g_minibatch = global_minibatch_number(epoch, 0, train_loader_len)
    tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, 0, float('inf'))
    tqdm_b_obj = tqdm(data_loader, desc=tqdm_b_descr)

    for batch_id, samples in enumerate(tqdm_b_obj):
        # prepare data before passing to model
        optimizer.zero_grad()
        batch_size = len(samples[0])
        frames_ = samples[1]
        frames = torch.stack(frames_).to(device)
        # print(f'{idx} | batch_size {batch_size} | frames:{frames.shape}')
        labels = torch.stack(samples[2]).type(torch.float).to(device)

        predictions = model(frames)

        predicted = torch.max(predictions.data, 1)[1]
        batch_corr = (predicted == labels).sum().item()

        predictions = torch.squeeze(predictions, dim=1)
        batch_loss = criterion(predictions, labels)

        total_samples += batch_size
        batch_loss_val = batch_loss.item()
        total_correct += batch_corr
        losses.append(batch_loss_val)
        batch_accuracy = batch_corr * 100 / batch_size
        accuracies.append(batch_accuracy)

        batch_loss.backward()
        optimizer.step()
        # print(f'predictions = {predicted}')
        # print(f'labels = {labels}')
        tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, batch_accuracy, batch_loss_val)
        g_minibatch = global_minibatch_number(epoch, batch_id, train_loader_len)
        # tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, np.mean(accuracies), np.mean(losses))
        tqdm_b_obj.set_description(tqdm_b_descr)
        tqdm_b_obj.update()

        sum_writer.add_scalar('Batch-loss', batch_loss_val, g_minibatch)
        sum_writer.add_scalar('Batch-accuracy', batch_accuracy, g_minibatch)

    mean_epoch_acc = np.mean(accuracies)
    mean_epoch_loss = np.mean(losses)
    # print(f'manual mean acc = {total_correct * 100 / total_samples},  mean api acc = {mean_epoch_acc}')

    return model, mean_epoch_acc, mean_epoch_loss


def valid_epoch(epoch=None, model=None, criterion=None, optimizer=None, data_loader=None, batch_size=None, device=None,
                log_dir=None, sum_writer=None):
    losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0
    valid_loader_len = len(data_loader)
    model.train(True)

    tqdm_b_descr_format = "Validation. Batch# {:05} Accuracy = {:02.4f}%, Loss = {:.8f}"
    g_minibatch = global_minibatch_number(epoch, 0, valid_loader_len)
    tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, 0, float('inf'))
    tqdm_b_obj = tqdm(data_loader, desc=tqdm_b_descr)

    model.eval()
    with torch.no_grad():
        for batch_id, samples in enumerate(tqdm_b_obj):
            # prepare data before passing to model
            batch_size = len(samples[0])
            frames_ = samples[1]
            frames = torch.stack(frames_).to(device)
            labels = torch.stack(samples[2]).type(torch.float).to(device)

            predictions = model(frames)

            predicted = torch.max(predictions.data, 1)[1]
            batch_corr = (predicted == labels).sum().item()

            predictions = torch.squeeze(predictions, dim=1)
            batch_loss = criterion(predictions, labels)

            total_samples += batch_size
            batch_loss_val = batch_loss.item()
            total_correct += batch_corr
            losses.append(batch_loss_val)
            batch_accuracy = batch_corr * 100 / batch_size
            accuracies.append(batch_accuracy)

            tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, batch_accuracy, batch_loss_val)
            g_minibatch = global_minibatch_number(epoch, batch_id, valid_loader_len)
            # tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, np.mean(accuracies), np.mean(losses))
            tqdm_b_obj.set_description(tqdm_b_descr)
            tqdm_b_obj.update()

            sum_writer.add_scalar('Batch-loss', batch_loss_val, g_minibatch)
            sum_writer.add_scalar('Batch-accuracy', batch_accuracy, g_minibatch)

        mean_epoch_acc = np.mean(accuracies)
        mean_epoch_loss = np.mean(losses)
        # print(f'manual mean acc = {total_correct * 100 / total_samples},  mean api acc = {mean_epoch_acc}')

    return mean_epoch_acc, mean_epoch_loss
