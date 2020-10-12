import torch
import sys
from utils import *
from data_utils.utils import *
from data_utils.datasets import DFDCDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.checkpoint import *
from quantification.utils import *
import cv2
import torchvision
from features.encoders import *
from models.utils import *


def test_model(model, model_params, criterion, log_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
    ])

    # num_workers = multiprocessing.cpu_count() - 2
    num_workers = 0

    test_dataset = DFDCDataset(test_data, mode='test', transform=test_transform,
                               max_num_frames=model_params['max_num_frames'],
                               frame_dim=imsize)

    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                             collate_fn=my_collate)

    print(f"Batch_size {model_params['batch_size']}")
    print(f'Test data len {test_data_len}')

    tqdm_test_descr_format = "Testing model: Test acc = {:02.4f}%, Mean train loss = {:.8f}"
    tqdm_test_descr = tqdm_test_descr_format.format(0, float('inf'))
    tqdm_test_obj = tqdm(test_loader, desc=tqdm_test_descr)

    losses = []
    accuracies = []
    all_video_filenames = []
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
            batch_size = len(samples[0])
            all_video_filenames.extend(samples[0])
            frames_ = samples[1]
            frames = torch.stack(frames_).to(device)
            labels = torch.stack(samples[2]).to(device)

            output = model(frames)

            batch_loss = criterion(output, labels)
            batch_loss_val = batch_loss.item()

            predicted = get_predictions(output)
            batch_corr = (predicted == labels).sum().item()
            all_predicted_labels.extend(predicted.tolist())
            all_ground_truth_labels.extend(labels.tolist())
            total_samples += batch_size
            total_correct += batch_corr
            losses.append(batch_loss_val)
            batch_accuracy = batch_corr * 100 / batch_size
            accuracies.append(batch_accuracy)

            tqdm_test_descr = tqdm_test_descr_format.format(batch_accuracy, batch_loss_val)
            tqdm_test_obj.set_description(tqdm_test_descr)
            tqdm_test_obj.update()

    report_type = 'Test'
    print(f'Saving model results for {report_type}')
    save_model_results_to_log(model=model, model_params=model_params,
                              losses=losses, accuracies=accuracies,
                              predicted=all_predicted_labels, ground_truth=all_ground_truth_labels,
                              sample_names=all_video_filenames,
                              log_dir=log_dir, report_type=report_type)
