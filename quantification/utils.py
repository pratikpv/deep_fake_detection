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


def save_model_results_to_log(model=None, model_params=None,
                              train_losses=None, train_accuracy=None,
                              predicted=None, ground_truth=None,
                              misc_data=None, sample_names=None, log_dir=None):
    print('Saving model results', end='')
    log_params = get_log_params()
    model_name = model_params['model_name']
    num_of_classes = 2
    # real = 0, fake = 1
    class_names = ['Real', 'Fake']

    model_log_dir = os.path.join(log_dir, model_name)
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

    sample_names_mp4 = [i + '.mp4' for i in sample_names]
    df = pd.DataFrame([sample_names_mp4, ground_truth, predicted]).T
    df.columns = ['sample_name', 'ground_truth', 'predictions']
    df = df.set_index(['sample_name'])
    df.to_csv(all_samples_pred_csv)

    # generate and save confusion matrix
    plot_x_label = "Predictions"
    plot_y_label = "Actual"
    cmap = plt.cm.Blues
    pred_class_indexes = sorted(np.unique(predicted))
    pred_num_classes = len(pred_class_indexes)
    target_class_names = [class_names[int(i)] for i in pred_class_indexes]

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

    if train_losses is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(train_losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(model_loss_png)
        plt.close(fig)

        # save model training stats
        with open(model_train_losses_log_file, 'wb') as file:
            pickle.dump(train_losses, file)
            file.flush()

    if train_accuracy is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(train_accuracy, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.savefig(model_accuracy_png)
        plt.close(fig)

        with open(model_train_accuracy_log_file, 'wb') as file:
            pickle.dump(train_accuracy, file)
            file.flush()

    report = metrics.classification_report(ground_truth, predicted, target_names=list(target_class_names))

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

        if misc_data:
            file.write('misc data: ' + misc_data + '\n')
            file.write('-' * log_params['line_len'] + '\n')

        file.write('classification report' + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write(report + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.flush()

    # save model as pytorch state dict
    torch.save(model.state_dict(), model_save_path)
    sys.stdout.flush()
