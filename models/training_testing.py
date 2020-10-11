import torch
import sys
from utils import *
from data_utils.utils import *
from models.DeepFakeDetectModel_1 import *
from models.DeepFakeDetectModel_2 import *
from models.DeepFakeDetectModel_3 import *
from models.DeepFakeDetectModel_4 import *
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
from models.checkpoint import *
from quantification.utils import *
import cv2
from models.utils import *

cv2.setNumThreads(0)


def train_model(train_method=None):
    if train_method is None:
        return None

    log_dir = get_log_dir_name()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_params = get_training_params()

    use_processed_data = True
    if use_processed_data:
        train_data = get_processed_training_video_filepaths()
        valid_data = get_processed_validation_video_filepaths()
    else:
        train_data = get_all_training_video_filepaths(get_train_data_path())
        valid_data = get_all_validation_video_filepaths(get_validation_data_path())

    train_data_len = len(train_data)
    valid_data_len = len(valid_data)

    sample_random = True
    train_size = get_training_sample_size()
    if train_size > 0.0:
        train_data_len = int(train_data_len * train_size)
        if sample_random:
            train_data = random.sample(train_data, train_data_len)
        else:
            train_data = train_data[0: train_data_len]

    valid_size = get_valid_sample_size()
    if valid_size > 0.0:
        valid_data_len = int(valid_data_len * valid_size)
        if sample_random:
            valid_data = random.sample(valid_data, valid_data_len)
        else:
            valid_data = valid_data[0: valid_data_len]

    encoder_name = get_default_cnn_encoder_name()
    imsize = encoder_params[encoder_name]["imsize"]

    train_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DFDCDataset(train_data, mode='train', transform=train_transform,
                                max_num_frames=model_params['max_num_frames'],
                                frame_dim=imsize)
    valid_dataset = DFDCDataset(valid_data, mode='valid', transform=valid_transform,
                                max_num_frames=model_params['max_num_frames'],
                                frame_dim=imsize)

    # num_workers = multiprocessing.cpu_count() - 2
    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                              shuffle=True, collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                              collate_fn=my_collate)

    print(f'Batch_size {train_loader.batch_size}')
    if model_params['model_name'] == 'DeepFakeDetectModel_2':
        model = DeepFakeDetectModel_2(frame_dim=imsize, max_num_frames=model_params['max_num_frames'],
                                      encoder_name=encoder_name).to(device)
    elif model_params['model_name'] == 'DeepFakeDetectModel_3':
        model = DeepFakeDetectModel_3(frame_dim=imsize).to(device)
    elif model_params['model_name'] == 'DeepFakeDetectModel_4':
        model = DeepFakeDetectModel_4(frame_dim=imsize, encoder_name=encoder_name).to(device)
    else:
        raise Exception("Unknown model name passed")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])

    print(f'Train data len {train_data_len}')
    print(f'Valid data len {valid_data_len}')
    start_epoch = 0
    if train_method == 'resume':
        start_epoch, model, optimizer = load_checkpoint(model, optimizer)
        print(f'Resuming Training from epoch {start_epoch}')
    else:
        print(f'Starting training from scratch')

    tqdm_train_descr_format = "Training model: Mean train acc = {:02.4f}%, Mean train loss = {:.8f} : Mean valid acc = {:02.4f}%, Mean valid loss = {:.8f}"
    tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'), 0, float('inf'))
    tqdm_train_obj = tqdm(range(model_params['epochs']), desc=tqdm_train_descr)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))

    lowest_v_epoch_loss = float('inf')
    model_train_accuracies = []
    model_train_losses = []
    model_valid_accuracies = []
    model_valid_losses = []

    for e in tqdm_train_obj:

        if e < start_epoch:
            continue

        train_valid_use_tqdm = False
        model, t_epoch_accuracy, t_epoch_loss = train_epoch(epoch=e, model=model, criterion=criterion,
                                                            optimizer=optimizer,
                                                            data_loader=train_loader,
                                                            batch_size=model_params['batch_size'],
                                                            device=device,
                                                            log_dir=log_dir, sum_writer=train_writer,
                                                            use_tqdm=train_valid_use_tqdm)
        model_train_accuracies.append(t_epoch_accuracy)
        model_train_losses.append(t_epoch_loss)

        tqdm_descr = tqdm_train_descr_format.format(np.mean(model_train_accuracies), np.mean(model_train_losses),
                                                    np.mean(model_valid_accuracies), np.mean(model_valid_losses))
        tqdm_train_obj.set_description(tqdm_descr)
        tqdm_train_obj.update()

        train_writer.add_scalar('Training: loss per epoch', t_epoch_loss, e)
        train_writer.add_scalar('Training: accuracy per epoch', t_epoch_accuracy, e)

        v_epoch_accuracy, v_epoch_loss, all_predicted_labels, \
        all_ground_truth_labels, all_video_filenames = valid_epoch(
            epoch=e, model=model, criterion=criterion, optimizer=optimizer,
            data_loader=valid_loader, batch_size=model_params['batch_size'],
            device=device,
            log_dir=log_dir, sum_writer=train_writer, use_tqdm=train_valid_use_tqdm)

        model_valid_accuracies.append(v_epoch_accuracy)
        model_valid_losses.append(v_epoch_loss)

        tqdm_descr = tqdm_train_descr_format.format(np.mean(model_train_accuracies), np.mean(model_train_losses),
                                                    np.mean(model_valid_accuracies), np.mean(model_valid_losses))
        tqdm_train_obj.set_description(tqdm_descr)
        tqdm_train_obj.update()

        train_writer.add_scalar('Validation: loss per epoch', v_epoch_loss, e)
        train_writer.add_scalar('Validation: accuracy per epoch', v_epoch_accuracy, e)

        save_checkpoint(epoch=e, model=model, optimizer=optimizer)
        if v_epoch_loss < lowest_v_epoch_loss:
            lowest_v_epoch_loss = v_epoch_loss

            save_model_results_to_log(model=model, model_params=model_params, train_losses=t_epoch_loss,
                                      train_accuracy=t_epoch_accuracy, predicted=all_predicted_labels,
                                      ground_truth=all_ground_truth_labels, sample_names=all_video_filenames,
                                      log_dir=log_dir)

    return model, model_params, criterion, log_dir


def train_epoch(epoch=None, model=None, criterion=None, optimizer=None, data_loader=None, batch_size=None, device=None,
                log_dir=None, sum_writer=None, use_tqdm=False):
    losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0
    train_loader_len = len(data_loader)
    model.train(True)

    train_data_iter = data_loader
    if use_tqdm:
        tqdm_b_descr_format = "Training. Batch# {:05} Accuracy = {:02.4f}%, Loss = {:.8f}"
        g_minibatch = global_minibatch_number(epoch, 0, train_loader_len)
        tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, 0, float('inf'))
        tqdm_b_obj = tqdm(data_loader, desc=tqdm_b_descr)
        train_data_iter = tqdm_b_obj

    # print(f'\ntraining epoch : {epoch}\n')
    for batch_id, samples in enumerate(train_data_iter):
        # prepare data before passing to model
        optimizer.zero_grad()
        batch_size = len(samples[0])
        frames_ = samples[1]
        frames = torch.stack(frames_).to(device)
        # print(f'{idx} | batch_size {batch_size} | frames:{frames.shape}')
        labels = torch.stack(samples[2]).to(device)

        output = model(frames)

        # print(f'train out= {output}')
        batch_loss = criterion(output, labels)
        batch_loss_val = batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        predicted = get_predictions(output)
        batch_corr = (predicted == labels).sum().item()

        # print(f'Train Predictions = {predicted}')
        # print(f'Train labels = {labels}')
        # print(f'Train batch_corr = {batch_corr}')

        total_samples += batch_size
        total_correct += batch_corr
        losses.append(batch_loss_val)
        batch_accuracy = batch_corr * 100 / batch_size
        accuracies.append(batch_accuracy)

        # g_minibatch = global_minibatch_number(epoch, batch_id, batch_size)
        if use_tqdm:
            tqdm_b_descr = tqdm_b_descr_format.format(batch_id, batch_accuracy, batch_loss_val)
            # tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, np.mean(accuracies), np.mean(losses))
            tqdm_b_obj.set_description(tqdm_b_descr)
            tqdm_b_obj.update()

        # sum_writer.add_scalar('Training: Batch-loss', batch_loss_val, g_minibatch)
        # sum_writer.add_scalar('Training: Batch-accuracy', batch_accuracy, g_minibatch)

    mean_epoch_acc = np.mean(accuracies)
    mean_epoch_loss = np.mean(losses)

    return model, mean_epoch_acc, mean_epoch_loss


def valid_epoch(epoch=None, model=None, criterion=None, optimizer=None, data_loader=None, batch_size=None, device=None,
                log_dir=None, sum_writer=None, use_tqdm=False):
    losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0
    valid_loader_len = len(data_loader)

    all_predicted_labels = []
    all_ground_truth_labels = []
    all_video_filenames = []

    valid_data_iter = data_loader
    if use_tqdm:
        tqdm_b_descr_format = "Validation. Batch# {:05} Accuracy = {:02.4f}%, Loss = {:.8f}"
        g_minibatch = global_minibatch_number(epoch, 0, valid_loader_len)
        tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, 0, float('inf'))
        tqdm_b_obj = tqdm(data_loader, desc=tqdm_b_descr)
        valid_data_iter = tqdm_b_obj

    model.eval()
    # print(f'\nvalid epoch : {epoch}')
    with torch.no_grad():
        for batch_id, samples in enumerate(valid_data_iter):
            # prepare data before passing to model
            batch_size = len(samples[0])
            all_video_filenames.extend(samples[0])
            frames_ = samples[1]
            frames = torch.stack(frames_).to(device)
            labels = torch.stack(samples[2]).to(device)

            output = model(frames)
            # print(f'valid out= {output}')
            batch_loss = criterion(output, labels)
            batch_loss_val = batch_loss.item()

            predicted = get_predictions(output)
            batch_corr = (predicted == labels).sum().item()
            all_predicted_labels.extend(predicted.tolist())
            all_ground_truth_labels.extend(labels.tolist())
            # print(f"Valid Predicted: {predicted}")
            # print(f"Valid Labels: {labels_list}")

            total_samples += batch_size
            total_correct += batch_corr
            losses.append(batch_loss_val)
            batch_accuracy = batch_corr * 100 / batch_size
            accuracies.append(batch_accuracy)

            # g_minibatch = global_minibatch_number(epoch, batch_id, batch_size)
            if use_tqdm:
                tqdm_b_descr = tqdm_b_descr_format.format(batch_id, batch_accuracy, batch_loss_val)
                # tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, np.mean(accuracies), np.mean(losses))
                tqdm_b_obj.set_description(tqdm_b_descr)
                tqdm_b_obj.update()

            # sum_writer.add_scalar('Validation: Batch-loss', batch_loss_val, g_minibatch)
            # sum_writer.add_scalar('Validation: Batch-accuracy', batch_accuracy, g_minibatch)

        mean_epoch_acc = np.mean(accuracies)
        mean_epoch_loss = np.mean(losses)

    return mean_epoch_acc, mean_epoch_loss, all_predicted_labels, all_ground_truth_labels, all_video_filenames
