import torch
import sys
from utils import *
from data_utils.utils import *
from data_utils.datasets import DFDCDataset, DFDCDatasetSimple
from torch.utils.data import DataLoader
import torch.nn as nn
import multiprocessing
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.checkpoint import *
from quantification.utils import *
import cv2
import torchvision
from features.encoders import *
from models.utils import *


def test_model(model, model_params, criterion, log_dir, model_kind):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if model_params['batch_format'] == 'stacked':
        use_processed_data = True
        if use_processed_data:
            test_data = get_processed_test_video_filepaths()
        else:
            test_data = get_all_test_video_filepaths(get_test_data_path())

        test_data_len = len(test_data)
        sample_random = True
        test_size = get_test_sample_size()
        if test_size > 0.0:
            test_data_len = int(test_data_len * test_size)
            if sample_random:
                test_data = random.sample(test_data, test_data_len)
            else:
                test_data = test_data[0: test_data_len]

    encoder_name = get_default_cnn_encoder_name()
    imsize = encoder_params[encoder_name]["imsize"]

    test_transform = torchvision.transforms.Compose([
        transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #num_workers = multiprocessing.cpu_count() - 2
    num_workers = 0

    if model_params['batch_format'] == 'stacked':
        test_dataset = DFDCDataset(test_data, mode='test', transform=test_transform,
                                   max_num_frames=model_params['max_num_frames'],
                                   frame_dim=model_params['imsize'])

        test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                                 collate_fn=my_collate)
    elif model_params['batch_format'] == 'simple':
        test_dataset = DFDCDatasetSimple(mode='test', transform=test_transform, data_size=get_test_sample_size(),
                                         dataset=model_params['dataset'])

        test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                                 pin_memory=True)
    else:
        raise Exception("model_params['batch_format'] not supported")

    print(f"Batch_size {model_params['batch_size']}")

    tqdm_test_descr_format = "Testing [Acc={:02.4f}% |Loss Total={:.8f}, Fake={:.8f}, Real={:.8f}]"

    tqdm_test_descr = tqdm_test_descr_format.format(0, float('inf'), float('inf'), float('inf'))
    tqdm_test_obj = tqdm(test_loader, desc=tqdm_test_descr)

    losses = []
    fake_losses = []
    real_losses = []
    accuracies = []
    probabilities = []
    all_filenames = []
    all_predicted_labels = []
    all_ground_truth_labels = []
    total_samples = 0
    total_correct = 0
    model.eval()
    model = model.to(device)
    criterion = criterion.to(device)
    with torch.no_grad():
        for batch_id, samples in enumerate(tqdm_test_obj):
            # prepare data before passing to model

            if model_params['batch_format'] == 'stacked':
                batch_size = len(samples[0])
                all_filenames.extend(samples[0])
                frames_ = samples[1]
                frames = torch.stack(frames_).to(device)
                labels = torch.stack(samples[2]).to(device).unsqueeze(1)
            elif model_params['batch_format'] == 'simple':
                frames = samples['frame_tensor'].to(device)
                labels = samples['label'].to(device).unsqueeze(1)
                batch_size = labels.shape[0]
                for i in range(batch_size):
                    all_filenames.append(str(samples['video_id'][i]) + '__' +
                                         str(samples['frame'][i]))
            else:
                raise Exception("model_params['batch_format'] not supported")

            output = model(frames)

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

            total_samples += batch_size
            total_correct += batch_corr
            losses.append(batch_loss_val)
            fake_losses.append(fake_loss_val)
            real_losses.append(real_loss_val)
            batch_accuracy = batch_corr * 100 / batch_size
            accuracies.append(batch_accuracy)
            if len(predicted) > 1:
                all_predicted_labels.extend(predicted.squeeze())
                all_ground_truth_labels.extend(labels.squeeze())
                probabilities.extend(class_probability.squeeze())
            else:
                all_predicted_labels.append(predicted.squeeze())
                all_ground_truth_labels.append(labels.squeeze())
                probabilities.append(class_probability.squeeze())

            tqdm_test_descr = tqdm_test_descr_format.format(batch_accuracy, batch_loss_val, fake_loss_val,
                                                            real_loss_val)
            tqdm_test_obj.set_description(tqdm_test_descr)
            tqdm_test_obj.update()

    report_type = 'Test'
    print(f'Saving model results for {report_type}')
    save_model_results_to_log(model=model, model_params=model_params,
                              losses=losses, accuracies=accuracies,
                              predicted=all_predicted_labels, ground_truth=all_ground_truth_labels,
                              sample_names=all_filenames,
                              log_dir=log_dir, log_kind=model_kind, report_type=report_type,
                              probabilities=probabilities)

    print_green(
        f'Test | Acc={np.mean(accuracies)}, Total Loss={np.mean(losses)}, Fake Loss={np.mean(fake_losses)}, Real Loss={np.mean(real_losses)}')
