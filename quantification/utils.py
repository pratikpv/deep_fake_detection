import os
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import sys
from utils import *
from models.checkpoint import *


def save_model_results_to_log(model=None, model_params=None, losses=None, accuracies=None, predicted=None,
                              ground_truth=None, misc_data=None, sample_names=None, log_dir=None, log_kind=None,
                              report_type=None, probabilities=None):
    log_params = get_log_params()
    model_name = model_params['model_name']
    num_of_classes = 2
    # real = 0, fake = 1
    class_names = ['Real', 'Fake']
    if log_kind:
        log_dir_kind = os.path.join(log_dir, log_kind)
    else:
        log_dir_kind = os.path.join(log_dir)
    model_log_dir = os.path.join(log_dir_kind, model_name, report_type)
    os.makedirs(model_log_dir, exist_ok=True)
    model_log_file = os.path.join(model_log_dir, log_params['model_info_log'])
    model_train_losses_log_file = os.path.join(model_log_dir, log_params['model_loss_info_log'])
    model_train_accuracy_log_file = os.path.join(model_log_dir, log_params['model_acc_info_log'])
    model_save_path = os.path.join(model_log_dir, model_name + '.pt')
    model_conf_mat_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_csv'])
    model_conf_mat_png = os.path.join(model_log_dir, log_params['model_conf_matrix_png'])
    model_conf_mat_normalized_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_csv'])
    model_conf_mat_normalized_png = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_png'])
    model_loss_png = os.path.join(model_log_dir, log_params['model_loss_png'])
    model_accuracy_png = os.path.join(model_log_dir, log_params['model_accuracy_png'])
    all_samples_pred_csv = os.path.join(model_log_dir, log_params['all_samples_pred_csv'])

    report = None
    if predicted is not None:
        df = pd.DataFrame([sample_names, ground_truth, predicted, probabilities]).T
        df.columns = ['sample_name', 'ground_truth', 'predictions', 'probability']
        df = df.set_index(['sample_name'])
        df.to_csv(all_samples_pred_csv)

        # generate and save confusion matrix
        plot_x_label = "Predictions"
        plot_y_label = "Actual"
        cmap = plt.cm.Blues
        # pred_class_indexes = sorted(np.unique(predicted))
        # pred_num_classes = len(pred_class_indexes)
        # target_class_names = [class_names[int(i)] for i in pred_class_indexes]

        cm = metrics.confusion_matrix(ground_truth, predicted)
        target_class_names = class_names  # TODO

        df_confusion = pd.DataFrame(cm)
        df_confusion.index = target_class_names
        df_confusion.columns = target_class_names
        df_confusion.round(2)
        df_confusion.to_csv(model_conf_mat_csv)
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(df_confusion, annot=True, cmap=cmap)
        plt.xlabel(plot_x_label)
        plt.ylabel(plot_y_label)
        plt.title('Confusion Matrix')
        plt.savefig(model_conf_mat_png)
        plt.close(fig)

        cm = metrics.confusion_matrix(ground_truth, predicted, normalize='all')
        df_confusion = pd.DataFrame(cm)
        df_confusion.index = target_class_names
        df_confusion.columns = target_class_names
        df_confusion.round(2)
        df_confusion.to_csv(model_conf_mat_normalized_csv)
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(df_confusion, annot=True, cmap=cmap)
        plt.xlabel(plot_x_label)
        plt.ylabel(plot_y_label)
        plt.title('Normalized Confusion Matrix')
        plt.savefig(model_conf_mat_normalized_png)
        plt.close(fig)

        report = metrics.classification_report(ground_truth, predicted, target_names=list(target_class_names))

    if losses is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(report_type + ' Loss')
        plt.legend()
        plt.savefig(model_loss_png)
        plt.close(fig)

        # save model training stats
        with open(model_train_losses_log_file, 'wb') as file:
            pickle.dump(losses, file)
            file.flush()

    if accuracies is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(report_type + ' Accuracy')
        plt.legend()
        plt.savefig(model_accuracy_png)
        plt.close(fig)

        with open(model_train_accuracy_log_file, 'wb') as file:
            pickle.dump(accuracies, file)
            file.flush()

    # save model arch and params
    with open(model_log_file, 'a') as file:
        file.write('-' * log_params['line_len'] + '\n')
        file.write('model architecture' + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write(str(model) + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write('model params' + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write(str(model_params) + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write('-' * log_params['line_len'] + '\n')

        if misc_data is not None:
            file.write('misc data: ' + misc_data + '\n')
            file.write('-' * log_params['line_len'] + '\n')

        if report is not None:
            file.write(report_type + ' classification report' + '\n')
            file.write('-' * log_params['line_len'] + '\n')
            file.write(report + '\n')
            file.write('-' * log_params['line_len'] + '\n')

        if report_type == 'Test':
            if losses is not None:
                file.write('Mean loss: ' + str(np.mean(losses)) + '\n')
                file.write('-' * log_params['line_len'] + '\n')

            if accuracies is not None:
                file.write('Mean accuracy:' + str(np.mean(accuracies)) + '\n')
                file.write('-' * log_params['line_len'] + '\n')

    copy_config(dest=model_log_dir)
    sys.stdout.flush()


def save_all_model_results(model=None, model_params=None, train_losses=None, train_accuracies=None, valid_losses=None,
                           valid_accuracies=None, valid_predicted=None, valid_ground_truth=None,
                           valid_sample_names=None, optimizer=None, criterion=None, epoch=0, log_dir=None,
                           log_kind=None, probabilities=None, amp_dict=None):
    report_type = 'Train'
    save_model_results_to_log(model=model, model_params=model_params,
                              losses=train_losses, accuracies=train_accuracies,
                              log_dir=log_dir, log_kind=log_kind, report_type=report_type)

    report_type = 'Validation'
    save_model_results_to_log(model=model, model_params=model_params,
                              losses=valid_losses, accuracies=valid_accuracies,
                              predicted=valid_predicted, ground_truth=valid_ground_truth,
                              sample_names=valid_sample_names,
                              log_dir=log_dir, log_kind=log_kind, report_type=report_type, probabilities=probabilities)

    save_checkpoint(epoch=epoch, model=model, model_params=model_params,
                    optimizer=optimizer, criterion=criterion.__class__.__name__, log_dir=log_dir, amp_dict=amp_dict)


def get_per_video_stat(df, vid, prob_threshold_fake, prob_threshold_real):
    # number of frames detected as fake with at-least prob of prob_threshold
    df1 = df.loc[df['video'] == vid]
    num_fake_frames = np.sum(
        np.array(df1[df1['predictions'] == 1]['norm_probability'].values) >= prob_threshold_fake)
    num_real_frames = np.sum(
        np.array(df1[df1['predictions'] == 0]['norm_probability'].values) >= prob_threshold_real)
    total_number_frames = len(df1)
    ground_truth = df1['ground_truth'].values[0]
    return num_fake_frames, num_real_frames, total_number_frames, ground_truth


def split_video(val):
    return val.split('__')[0]


def split_frames(val):
    return val.split('__')[1]


def norm_probability(pred, prob):
    if pred == 0:
        return 1 - prob
    else:
        return prob


def pred_strategy(num_fake_frames, num_real_frames, total_number_frames):
    if num_fake_frames >= (0.10 * total_number_frames):
        return 1
    return 0


def gen_report_for_per_frame_model(per_frame_csv=None, log_dir=None, report_type=None, prob_threshold_fake=0.50,
                                   prob_threshold_real=0.55):
    df = pd.read_csv(per_frame_csv)

    df['video'] = df['sample_name'].apply(split_video)
    df['frames'] = df['sample_name'].apply(split_frames)
    df['norm_probability'] = df.apply(lambda x: norm_probability(x.predictions, x.probability), axis=1)
    all_videos = set(df['video'].values)

    final_df = pd.DataFrame(
        columns=['video', 'num_fake_frames', 'num_real_frames', 'total_number_frames', 'ground_truth'])

    for v in all_videos:
        num_fake_frames, num_real_frames, total_number_frames, \
        ground_truth = get_per_video_stat(df, v, prob_threshold_fake, prob_threshold_real)
        prediction = pred_strategy(num_fake_frames, num_real_frames, total_number_frames)
        final_df = final_df.append({'video': v, 'num_fake_frames': num_fake_frames,
                                    'num_real_frames': num_real_frames, 'total_number_frames': total_number_frames,
                                    'ground_truth': ground_truth, 'prediction': prediction},
                                   ignore_index=True)

    log_params = get_log_params()
    model_log_dir = os.path.join(log_dir, 'final', report_type)
    os.makedirs(model_log_dir, exist_ok=True)

    model_conf_mat_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_csv'])
    model_conf_mat_png = os.path.join(model_log_dir, log_params['model_conf_matrix_png'])
    model_conf_mat_normalized_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_csv'])
    model_conf_mat_normalized_png = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_png'])
    all_samples_pred_csv = os.path.join(model_log_dir, log_params['all_samples_pred_csv'])
    model_log_file = os.path.join(model_log_dir, log_params['model_info_log'])

    final_df = final_df.set_index(['video'])
    final_df.to_csv(all_samples_pred_csv)

    # generate and save confusion matrix
    plot_x_label = "Predictions"
    plot_y_label = "Actual"
    cmap = plt.cm.Blues
    class_names = ['Real', 'Fake']

    cm = metrics.confusion_matrix(final_df['ground_truth'], final_df['prediction'])
    target_class_names = class_names

    df_confusion = pd.DataFrame(cm)
    df_confusion.index = target_class_names
    df_confusion.columns = target_class_names
    df_confusion.round(2)
    df_confusion.to_csv(model_conf_mat_csv)
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(df_confusion, annot=True, cmap=cmap)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)
    plt.title('Confusion Matrix')
    plt.savefig(model_conf_mat_png)
    plt.close(fig)

    cm = metrics.confusion_matrix(final_df['ground_truth'], final_df['prediction'], normalize='all')
    df_confusion = pd.DataFrame(cm)
    df_confusion.index = target_class_names
    df_confusion.columns = target_class_names
    df_confusion.round(2)
    df_confusion.to_csv(model_conf_mat_normalized_csv)
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(df_confusion, annot=True, cmap=cmap)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(model_conf_mat_normalized_png)
    plt.close(fig)

    report = metrics.classification_report(final_df['ground_truth'], final_df['prediction'],
                                           target_names=list(target_class_names))

    with open(model_log_file, 'a') as file:
        if report is not None:
            file.write(report_type + ' classification report' + '\n')
            file.write('-' * log_params['line_len'] + '\n')
            file.write(report + '\n')
            file.write('-' * log_params['line_len'] + '\n')
