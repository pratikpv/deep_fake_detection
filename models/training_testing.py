import torch
import sys
from utils import *
from data_utils.utils import *
from apex import amp
from data_utils.datasets import DFDCDataset, DFDCDatasetSimple
from torch.utils.data import DataLoader
import torch.nn as nn
import multiprocessing
import numpy as np
import torchvision
from features.encoders import *
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from quantification.utils import *
import cv2
from models.utils import *
import PIL
from pathlib import Path

cv2.setNumThreads(0)


def train_model(log_dir=None, train_resume_checkpoint=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_params = get_training_params()

    #
    # model_params['batch_format'] = 'stacked' -> generates batches as below
    # batch_size x number_of_frames_stacked x color_channel x frame_height x frame-width
    # in which number_of_frames_stacked is Y number of frames stacked. Y can be configured with max_num_frames
    #
    # model_params['batch_format'] = 'simple' -> generates batches as below
    # batch_size x color_channel x frame_height x frame-width
    #
    if model_params['batch_format'] == 'stacked':
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
    model_params['encoder_name'] = encoder_name
    model_params['imsize'] = imsize

    def gaussian_blur(img):
        return img.filter(PIL.ImageFilter.BoxBlur(random.choice([1, 2, 3])))

    if model_params['train_transform'] == 'complex':
        train_transform = torchvision.transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomCrop(imsize, imsize),
                transforms.ColorJitter(contrast=random.random()),
                transforms.Lambda(gaussian_blur),
            ]),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomGrayscale(p=0.05),
                transforms.ColorJitter(brightness=random.random()),
                transforms.RandomRotation(30),
            ]),
            transforms.Resize((imsize, imsize)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(0.30),
        ])

    elif model_params['train_transform'] == 'simple':
        train_transform = torchvision.transforms.Compose([
            transforms.Resize((imsize, imsize)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(0.30),
        ])
    else:
        raise Exception("model_params['train_transform'] not supported")

    valid_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # num_workers = multiprocessing.cpu_count() - 2
    num_workers = 0

    if model_params['batch_format'] == 'stacked':
        train_dataset = DFDCDataset(train_data, mode='train', transform=train_transform,
                                    max_num_frames=model_params['max_num_frames'],
                                    frame_dim=model_params['imsize'], random_sorted=model_params['random_sorted'],
                                    expand_label_dim=model_params['expand_label_dim'],
                                    fill_empty=model_params['fill_empty'],
                                    label_smoothing=model_params['label_smoothing'])
        valid_dataset = DFDCDataset(valid_data, mode='valid', transform=valid_transform,
                                    max_num_frames=model_params['max_num_frames'],
                                    frame_dim=model_params['imsize'], random_sorted=model_params['random_sorted'],
                                    expand_label_dim=model_params['expand_label_dim'],
                                    fill_empty=model_params['fill_empty'])

        train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                                  shuffle=True, collate_fn=my_collate)
        valid_loader = DataLoader(valid_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                                  collate_fn=my_collate)

    elif model_params['batch_format'] == 'simple':
        train_dataset = DFDCDatasetSimple(mode='train', transform=train_transform, data_size=get_training_sample_size(),
                                          dataset=model_params['dataset'],
                                          label_smoothing=model_params['label_smoothing'])
        valid_dataset = DFDCDatasetSimple(mode='valid', transform=valid_transform, data_size=get_valid_sample_size(),
                                          dataset=model_params['dataset'])

        train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                                  shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                                  shuffle=True, pin_memory=True)
    else:
        raise Exception("model_params['batch_format'] not supported")

    print(f"Batch_size {model_params['batch_size']}")
    model = get_model(model_params).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])

    if model_params['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=model_params['opt_level'],
                                          loss_scale='dynamic')
    print(f'model params {model_params}')
    start_epoch = 0
    lowest_v_epoch_loss = float('inf')
    highest_v_epoch_acc = 0.0
    model_train_accuracies = []
    model_train_losses = []
    model_valid_accuracies = []
    model_valid_losses = []

    if train_resume_checkpoint is not None:
        saved_epoch, model, optimizer, _, log_dir, amp_dict = load_checkpoint(model, optimizer,
                                                                              train_resume_checkpoint)
        if model_params['fp16']:
            amp.load_state_dict(amp_dict)
        start_epoch = saved_epoch + 1
        print(f'Resuming Training from epoch {start_epoch}')
        if os.path.basename(log_dir) == 'highest_acc' or os.path.basename(log_dir) == 'lowest_loss':
            log_dir = Path(log_dir).parent

        print(f'Resetting log_dir to {log_dir}')

        model_train_accuracies, model_train_losses, model_valid_accuracies, \
        model_valid_losses = load_acc_loss(model, log_dir)

        if len(model_train_accuracies) != start_epoch:
            raise Exception(
                f'Error! model_train_accuracies = {model_train_accuracies} len is {len(model_train_accuracies)}, expected {start_epoch}')
        if len(model_train_losses) != start_epoch:
            raise Exception(f'Error! model_train_losses = {model_train_losses}')
        if len(model_valid_accuracies) != start_epoch:
            raise Exception(f'Error! model_valid_accuracies = {model_valid_accuracies}')
        if len(model_valid_losses) != start_epoch:
            raise Exception(f'Error! model_valid_losses = {model_valid_losses}')
        lowest_v_epoch_loss = min(model_valid_losses)
        highest_v_epoch_acc = max(model_valid_accuracies)
        print(f'Loaded model acc and losses data successfully')

    else:
        print(f'Starting training from scratch')

    tqdm_train_descr_format = "Training[Train Acc={:02.4f}%|Loss Total={:.8f} Fake={:.8f} Real={:.8f}][Val Acc={:02.4f}%|Loss Total={:.8f} Fake={:.8f} Real={:.8f}]"
    tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'), float('inf'), float('inf'), 0, float('inf'),
                                                      float('inf'), float('inf'))
    tqdm_train_obj = tqdm(range(model_params['epochs']), desc=tqdm_train_descr)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))

    for e in tqdm_train_obj:

        if e < start_epoch:
            print(f"Skipping epoch {e}")
            continue

        train_valid_use_tqdm = True
        model, t_epoch_accuracy, t_epoch_loss, \
        t_epoch_fake_loss, t_epoch_real_loss = train_epoch(epoch=e, model=model,
                                                           criterion=criterion,
                                                           optimizer=optimizer,
                                                           data_loader=train_loader,
                                                           batch_size=model_params['batch_size'],
                                                           device=device,
                                                           log_dir=log_dir,
                                                           sum_writer=train_writer,
                                                           use_tqdm=train_valid_use_tqdm,
                                                           model_params=model_params)
        model_train_accuracies.append(t_epoch_accuracy)
        model_train_losses.append(t_epoch_loss)

        tqdm_descr = tqdm_train_descr_format.format(t_epoch_accuracy, t_epoch_loss,
                                                    t_epoch_fake_loss, t_epoch_real_loss,
                                                    0, float('inf'), float('inf'), float('inf'))
        tqdm_train_obj.set_description(tqdm_descr)
        tqdm_train_obj.update()

        train_writer.add_scalar('Training: loss per epoch', t_epoch_loss, e)
        train_writer.add_scalar('Training: fake loss per epoch', t_epoch_fake_loss, e)
        train_writer.add_scalar('Training: real loss per epoch', t_epoch_real_loss, e)
        train_writer.add_scalar('Training: accuracy per epoch', t_epoch_accuracy, e)

        print_green(tqdm_descr)
        v_epoch_accuracy, v_epoch_loss, v_epoch_fake_loss, \
        v_epoch_real_loss, all_predicted_labels, \
        all_ground_truth_labels, all_filenames, \
        probabilities = valid_epoch(epoch=e, model=model,
                                    criterion=criterion,
                                    data_loader=valid_loader,
                                    batch_size=model_params['batch_size'],
                                    device=device,
                                    log_dir=log_dir,
                                    sum_writer=train_writer,
                                    use_tqdm=train_valid_use_tqdm,
                                    model_params=model_params)

        model_valid_accuracies.append(v_epoch_accuracy)
        model_valid_losses.append(v_epoch_loss)

        tqdm_descr = tqdm_train_descr_format.format(t_epoch_accuracy, t_epoch_loss, t_epoch_fake_loss,
                                                    t_epoch_real_loss,
                                                    v_epoch_accuracy, v_epoch_loss, v_epoch_fake_loss,
                                                    v_epoch_real_loss)
        tqdm_train_obj.set_description(tqdm_descr)
        tqdm_train_obj.update()

        train_writer.add_scalar('Validation: loss per epoch', v_epoch_loss, e)
        train_writer.add_scalar('Validation: fake loss per epoch', v_epoch_fake_loss, e)
        train_writer.add_scalar('Validation: real loss per epoch', v_epoch_real_loss, e)
        train_writer.add_scalar('Validation: accuracy per epoch', v_epoch_accuracy, e)

        print_green(tqdm_descr)
        print(f'Saving model results at {log_dir}/latest_epoch for epoch {e}')
        if model_params['fp16']:
            amp_dict = amp.state_dict()
        else:
            amp_dict = None
        save_all_model_results(model=model, model_params=model_params,
                               optimizer=optimizer, criterion=criterion.__class__.__name__,
                               train_losses=model_train_losses, train_accuracies=model_train_accuracies,
                               valid_losses=model_valid_losses, valid_accuracies=model_valid_accuracies,
                               valid_predicted=all_predicted_labels, valid_ground_truth=all_ground_truth_labels,
                               valid_sample_names=all_filenames,
                               epoch=e, log_dir=log_dir,  log_kind='latest_epoch', probabilities=probabilities,
                               amp_dict=amp_dict)

        if v_epoch_loss < lowest_v_epoch_loss:
            lowest_v_epoch_loss = v_epoch_loss
            print_green(f'Saving best model (low loss) results at {log_dir}/lowest_loss for epoch {e}')
            if model_params['fp16']:
                amp_dict = amp.state_dict()
            else:
                amp_dict = None
            save_all_model_results(model=model, model_params=model_params,
                                   optimizer=optimizer, criterion=criterion.__class__.__name__,
                                   train_losses=model_train_losses, train_accuracies=model_train_accuracies,
                                   valid_losses=model_valid_losses, valid_accuracies=model_valid_accuracies,
                                   valid_predicted=all_predicted_labels, valid_ground_truth=all_ground_truth_labels,
                                   valid_sample_names=all_filenames,
                                   epoch=e, log_dir=log_dir, log_kind='lowest_loss', probabilities=probabilities,
                                   amp_dict=amp_dict)

        if highest_v_epoch_acc < v_epoch_accuracy:
            highest_v_epoch_acc = v_epoch_accuracy
            print_green(f'Saving best model (high acc) results at {log_dir}/highest_acc for epoch {e}')
            if model_params['fp16']:
                amp_dict = amp.state_dict()
            else:
                amp_dict = None
            save_all_model_results(model=model, model_params=model_params,
                                   optimizer=optimizer, criterion=criterion.__class__.__name__,
                                   train_losses=model_train_losses, train_accuracies=model_train_accuracies,
                                   valid_losses=model_valid_losses, valid_accuracies=model_valid_accuracies,
                                   valid_predicted=all_predicted_labels, valid_ground_truth=all_ground_truth_labels,
                                   valid_sample_names=all_filenames,
                                   epoch=e, log_dir=log_dir, log_kind='highest_acc', probabilities=probabilities,
                                   amp_dict=amp_dict)

    return model, model_params, criterion, log_dir


def train_epoch(epoch=None, model=None, criterion=None, optimizer=None, data_loader=None, batch_size=None, device=None,
                log_dir=None, sum_writer=None, use_tqdm=False, model_params=None):
    losses = []
    fake_losses = []
    real_losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0
    train_loader_len = len(data_loader)
    model.train(True)

    train_data_iter = data_loader
    if use_tqdm:
        tqdm_b_descr_format = "Training. Batch# {:05} Acc={:02.4f}%|Loss Total={:.8f} Fake={:.8f} Real={:.8f}"
        g_minibatch = global_minibatch_number(epoch, 0, train_loader_len)
        tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, 0, float('inf'), float('inf'), float('inf'))
        tqdm_b_obj = tqdm(data_loader, desc=tqdm_b_descr)
        train_data_iter = tqdm_b_obj

    # print(f'\ntraining epoch : {epoch}\n')
    for batch_id, samples in enumerate(train_data_iter):
        # prepare data before passing to model
        optimizer.zero_grad()

        if model_params['batch_format'] == 'stacked':
            batch_size = len(samples[0])
            frames_ = samples[1]
            frames = torch.stack(frames_).to(device)
            # print(f'{idx} | batch_size {batch_size} | frames:{frames.shape}')
            labels = torch.stack(samples[2]).to(device)
        elif model_params['batch_format'] == 'simple':
            frames = samples['frame_tensor'].to(device)
            # print(f'{idx} | batch_size {batch_size} | frames:{frames.shape}')
            labels = samples['label'].to(device).unsqueeze(1)
            batch_size = labels.shape[0]
        else:
            raise Exception("model_params['batch_format'] not supported")

        output = model(frames)
        # print(f'train out= {output}')
        labels = labels.type_as(output)
        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5
        if torch.sum(fake_idx * 1) > 0:
            fake_loss = criterion(output[fake_idx], labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = criterion(output[real_idx], labels[real_idx])

        batch_loss = (fake_loss + real_loss) / 2

        batch_loss_val = batch_loss.item()
        real_loss_val = 0 if real_loss == 0 else real_loss.item()
        fake_loss_val = 0 if fake_loss == 0 else fake_loss.item()

        if model_params['fp16']:
            with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward()

        optimizer.step()

        predicted = get_predictions(output).to('cpu').detach().numpy()
        labels = labels.to('cpu').detach().numpy()
        if model_params['label_smoothing'] != 0:
            labels = labels.round()
        batch_corr = (predicted == labels).sum().item()

        # print(f'Train Predictions = {predicted}')
        # print(f'Train labels = {labels}')
        # print(f'Train batch_corr = {batch_corr}')

        total_samples += batch_size
        total_correct += batch_corr
        losses.append(batch_loss_val)
        fake_losses.append(fake_loss_val)
        real_losses.append(real_loss_val)

        batch_accuracy = batch_corr * 100 / batch_size
        accuracies.append(batch_accuracy)

        # g_minibatch = global_minibatch_number(epoch, batch_id, batch_size)
        if use_tqdm:
            tqdm_b_descr = tqdm_b_descr_format.format(batch_id, batch_accuracy, batch_loss_val, fake_loss_val,
                                                      real_loss_val)
            # tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, np.mean(accuracies), np.mean(losses))
            tqdm_b_obj.set_description(tqdm_b_descr)
            tqdm_b_obj.update()

        # sum_writer.add_scalar('Training: Batch-loss', batch_loss_val, g_minibatch)
        # sum_writer.add_scalar('Training: Batch-accuracy', batch_accuracy, g_minibatch)

    mean_epoch_acc = np.mean(accuracies)
    mean_epoch_loss = np.mean(losses)
    mean_epoch_fake_loss = np.mean(fake_losses)
    mean_epoch_real_loss = np.mean(real_losses)

    return model, mean_epoch_acc, mean_epoch_loss, mean_epoch_fake_loss, mean_epoch_real_loss


def valid_epoch(epoch=None, model=None, criterion=None, data_loader=None, batch_size=None, device=None,
                log_dir=None, sum_writer=None, use_tqdm=False, model_params=None):
    losses = []
    fake_losses = []
    real_losses = []
    accuracies = []
    probabilities = []
    total_samples = 0
    total_correct = 0
    valid_loader_len = len(data_loader)
    model.eval()

    all_predicted_labels = []
    all_ground_truth_labels = []
    all_filenames = []

    valid_data_iter = data_loader
    if use_tqdm:
        tqdm_b_descr_format = "Validation. Batch# {:05} Acc={:02.4f}%|Loss Total={:.8f} Fake={:.8f} Real={:.8f}"
        g_minibatch = global_minibatch_number(epoch, 0, valid_loader_len)
        tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, 0, float('inf'), float('inf'), float('inf'))
        tqdm_b_obj = tqdm(data_loader, desc=tqdm_b_descr)
        valid_data_iter = tqdm_b_obj

    # print(f'\nvalid epoch : {epoch}')
    with torch.no_grad():
        for batch_id, samples in enumerate(valid_data_iter):
            # prepare data before passing to model

            if model_params['batch_format'] == 'stacked':

                batch_size = len(samples[0])
                all_filenames.extend(samples[0])

                frames_ = samples[1]
                frames = torch.stack(frames_).to(device)
                labels = torch.stack(samples[2]).to(device)

            elif model_params['batch_format'] == 'simple':

                frames = samples['frame_tensor'].to(device)
                # print(f'{idx} | batch_size {batch_size} | frames:{frames.shape}')
                labels = samples['label'].to(device).unsqueeze(1)
                batch_size = labels.shape[0]
                for i in range(batch_size):
                    all_filenames.append(str(samples['video_id'][i].item()) + '__' +
                                         str(samples['frame'][i]))
            else:
                raise Exception("model_params['batch_format'] not supported")

            output = model(frames)
            # print(f'valid out= {output}')
            labels = labels.type_as(output)
            fake_loss = 0
            real_loss = 0
            fake_idx = labels > 0.5
            real_idx = labels <= 0.5
            if torch.sum(fake_idx * 1) > 0:
                fake_loss = criterion(output[fake_idx], labels[fake_idx])
            if torch.sum(real_idx * 1) > 0:
                real_loss = criterion(output[real_idx], labels[real_idx])

            batch_loss = (fake_loss + real_loss) / 2
            batch_loss_val = batch_loss.item()
            real_loss_val = 0 if real_loss == 0 else real_loss.item()
            fake_loss_val = 0 if fake_loss == 0 else fake_loss.item()

            predicted = get_predictions(output).to('cpu').detach().numpy()
            class_probability = get_probability(output).to('cpu').detach().numpy()

            labels = labels.to('cpu').detach().numpy()
            batch_corr = (predicted == labels).sum().item()

            all_predicted_labels.extend(predicted.squeeze())
            all_ground_truth_labels.extend(labels.squeeze())
            # print(f"Valid Predicted: {predicted}")
            # print(f"Valid Labels: {labels_list}")

            total_samples += batch_size
            total_correct += batch_corr
            losses.append(batch_loss_val)
            fake_losses.append(fake_loss_val)
            real_losses.append(real_loss_val)
            batch_accuracy = batch_corr * 100 / batch_size
            accuracies.append(batch_accuracy)
            probabilities.extend(class_probability.squeeze())

            # g_minibatch = global_minibatch_number(epoch, batch_id, batch_size)
            if use_tqdm:
                tqdm_b_descr = tqdm_b_descr_format.format(batch_id, batch_accuracy, batch_loss_val, fake_loss_val,
                                                          real_loss_val)
                # tqdm_b_descr = tqdm_b_descr_format.format(g_minibatch, np.mean(accuracies), np.mean(losses))
                tqdm_b_obj.set_description(tqdm_b_descr)
                tqdm_b_obj.update()

            # sum_writer.add_scalar('Validation: Batch-loss', batch_loss_val, g_minibatch)
            # sum_writer.add_scalar('Validation: Batch-accuracy', batch_accuracy, g_minibatch)

        mean_epoch_acc = np.mean(accuracies)
        mean_epoch_loss = np.mean(losses)
        mean_epoch_fake_loss = np.mean(fake_losses)
        mean_epoch_real_loss = np.mean(real_losses)

    return mean_epoch_acc, mean_epoch_loss, mean_epoch_fake_loss, mean_epoch_real_loss, all_predicted_labels, all_ground_truth_labels, all_filenames, probabilities
