import multiprocessing
from functools import partial
from tqdm import tqdm
from data_utils.utils import *
import data_utils.augmentation as augmentation
import data_utils.distractions as distractions
import data_utils.face_detection as fd
import torch
import os
import cv2
from utils import *
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import traceback
import argparse
import numpy as np
from data_utils.datasets import DFDCDataset
from torch.utils.data import DataLoader


def test_data_augmentation(input_file, output_folder):
    """
    Test all kinds of data augmentation. Applies supported augmentation to input_file
    and save individual output videos in output_folder
    :param input_file:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    augmentation_methods = augmentation.get_supported_augmentation_methods()
    augmentation_methods.remove('noise')
    augmentation_param = [augmentation.get_augmentation_setting_by_type(m) for m in augmentation_methods]
    noise_methods = augmentation.get_supported_noise_types()
    noise_methods_param = [augmentation.get_noise_param_setting(m) for m in noise_methods]
    augmentation_methods.extend(['noise'] * len(noise_methods))
    augmentation_param.extend(noise_methods_param)

    augmentation_plan = list(zip(augmentation_methods, augmentation_param))

    out_id = os.path.splitext(os.path.basename(input_file))[0]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for aug in augmentation_plan:
            aug_func, aug_param = aug
            if aug_func == 'noise':
                suffix = out_id + '_' + aug_func + '_' + aug_param['noise_type']
            else:
                suffix = out_id + '_' + aug_func
            outfile = os.path.join(output_folder, suffix + '.mp4')
            jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                         (input_file, outfile,),
                                         dict(augmentation=aug_func, augmentation_param=aug_param)
                                         )
                        )

        for job in tqdm(jobs, desc="Applying augmentation"):
            results.append(job.get())


def test_data_distraction(input_file, output_folder):
    """
    Test all kinds of data distraction. Applies supported distraction to input_file
    and save individual output videos in output_folder
    :param input_file:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    distraction_methods = distractions.get_supported_distraction_methods()
    distraction_params = [distractions.get_distractor_setting_by_type(m) for m in distraction_methods]
    distraction_plan = list(zip(distraction_methods, distraction_params))
    out_id = os.path.splitext(os.path.basename(input_file))[0]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for distract in distraction_plan:
            distract_func, distract_param = distract
            outfile = os.path.join(output_folder, out_id + '_' + distract_func + '.mp4')
            jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                         (input_file, outfile,),
                                         dict(distraction=distract_func, distraction_param=distract_param)
                                         ))

        for job in tqdm(jobs, desc="Applying distraction"):
            results.append(job.get())

    return results


def locate_faces_batch(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    input_filepath_list = glob(input_root + '/*.mp4')
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for input_filepath in input_filepath_list:
            output_filepath = os.path.join(output_root, os.path.basename(input_filepath))
            jobs.append(pool.apply_async(augmentation.locate_face_in_videofile,
                                         (input_filepath, output_filepath,),
                                         )
                        )

        for job in tqdm(jobs, desc="Tracking faces"):
            results.append(job.get())


def generate_poster(org_file, aug_output_folder, frames_count=10, out_res=None):
    id = os.path.splitext(os.path.basename(org_file))[0]
    org_out_images = os.path.join(aug_output_folder, id + '_original')
    extract_images_from_video(org_file, org_out_images)
    sub_fol = []
    for f in glob(aug_output_folder + '/*'):
        if os.path.isdir(f):
            sub_fol.append(f)
    sub_fol_count = len(sub_fol)
    # print(f'sub_fol_count {sub_fol_count}')
    result_grid_filename = os.path.join(aug_output_folder, 'poster.jpg')

    total_frames = 300
    step = int(total_frames / frames_count)
    images_list = ['{}.jpg'.format(i) for i in range(0, total_frames, step)]
    images_count = len(images_list)
    # print('Images: ', images_list)
    # print('Images count: ', images_count)

    out_res_per_image = 1920, 1080
    result_figsize_resolution = out_res_per_image[0] * images_count, out_res_per_image[1] * images_count
    # print(f'result_figsize_resolution = {result_figsize_resolution}')
    fig, axes = plt.subplots(sub_fol_count, frames_count,
                             figsize=(40, 40)
                             )

    fol_n_replace_str = id + '_'
    for i, folder in enumerate(sub_fol):
        fol_n = folder.replace(fol_n_replace_str, '')
        for j, frame in enumerate(images_list):
            plt_image = plt.imread(folder + '/' + frame)
            axes[i, j].imshow(plt_image)
            axes[i, j].axis('off')
            if i == 0:
                fnum = frame.replace('.jpg', '')
                axes[0, j].set_title('frame {}'.format(fnum))
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
    plt.savefig(result_grid_filename)

    print(f'generated {result_grid_filename}')


def generate_data_augmentation_plan(root_dir=None, plan_pkl_file=None, plan_txt_file=None):
    # Apply various kinds of data augmentation to 30 % of whole training set.
    # Sample without replacement in this case and each below case.
    # Form these randomly selected video files,
    #
    # apply distractions to 35% of files
    # distractions and random noise to 35%
    # distractions, random noise, and augmentation to 15%
    # noise to 5%
    # augmentation and noise to 5%
    # augmentation to 5%
    #

    results = get_all_training_video_filepaths(root_dir=root_dir)
    polulation_size = len(results)
    random.shuffle(results)

    samples_size = int(polulation_size * 0.30)
    distr_samples_size = int(samples_size * 0.35)
    dist_noise_sample_size = int(samples_size * 0.35)
    dist_noise_aug_size = int(samples_size * 0.15)
    noise_sample_size = int(samples_size * 0.05)
    aug_noise_sample_size = int(samples_size * 0.05)
    aug_sample_size = int(samples_size * 0.05)

    print(f'Total data count {polulation_size}')
    print(f'Total samples count {samples_size}')

    samples = random.sample(results, samples_size)

    distr_samples = random.sample(samples, distr_samples_size)
    samples = list(filter(lambda i: i not in distr_samples, samples))

    dist_noise_samples = random.sample(samples, dist_noise_sample_size)
    samples = list(filter(lambda i: i not in dist_noise_samples, samples))

    dist_noise_aug_samples = random.sample(samples, dist_noise_aug_size)
    samples = list(filter(lambda i: i not in dist_noise_aug_samples, samples))

    noise_samples = random.sample(samples, noise_sample_size)
    samples = list(filter(lambda i: i not in noise_samples, samples))

    aug_noise_samples = random.sample(samples, aug_noise_sample_size)
    samples = list(filter(lambda i: i not in aug_noise_samples, samples))

    aug_samples = random.sample(samples, aug_sample_size)

    assert len(distr_samples) == distr_samples_size
    assert len(dist_noise_samples) == dist_noise_sample_size
    assert len(dist_noise_aug_samples) == dist_noise_aug_size
    assert len(noise_samples) == noise_sample_size
    assert len(aug_noise_samples) == aug_noise_sample_size
    assert len(aug_samples) == aug_sample_size

    distr_samples_exec_plan = []
    for i in distr_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        entry = i, plan
        distr_samples_exec_plan.append(entry)
    # pprint(distr_samples_exec_plan)

    dist_noise_samples_exec_plan = []
    for i in dist_noise_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        dist_noise_samples_exec_plan.append(entry)
    # pprint(dist_noise_samples_exec_plan)

    dist_noise_aug_exec_plan = []
    for i in dist_noise_aug_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        entry = i, plan
        dist_noise_aug_exec_plan.append(entry)
    # pprint(dist_noise_aug_exec_plan)

    noise_samples_exe_plan = []
    for i in noise_samples:
        plan = list()
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        noise_samples_exe_plan.append(entry)
    # pprint(noise_samples_exe_plan)

    aug_noise_samples_exec_plan = []
    for i in aug_noise_samples:
        plan = list()
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        aug_noise_samples_exec_plan.append(entry)
    # pprint(aug_noise_samples_exec_plan)

    aug_samples_exec_plan = []
    for i in aug_samples:
        plan = list()
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        entry = i, plan
        aug_samples_exec_plan.append(entry)
    # pprint(aug_samples_exec_plan)

    data_augmentation_plan = list()
    data_augmentation_plan.extend(distr_samples_exec_plan)
    data_augmentation_plan.extend(dist_noise_samples_exec_plan)
    data_augmentation_plan.extend(dist_noise_aug_exec_plan)
    data_augmentation_plan.extend(noise_samples_exe_plan)
    data_augmentation_plan.extend(aug_noise_samples_exec_plan)
    data_augmentation_plan.extend(aug_samples_exec_plan)

    print(f'Saving plan to {plan_pkl_file}')
    with open(plan_pkl_file, 'wb') as f:
        pickle.dump(data_augmentation_plan, f)

    print(f'Saving plan to {plan_txt_file}')
    with open(plan_txt_file, 'w') as f:
        for listitem in data_augmentation_plan:
            f.write('%s\n' % str(listitem))

    return data_augmentation_plan


def check_plan_already_executed(filename, plan, metadata_folder):
    id = os.path.basename(filename)
    file_search = metadata_folder + '/{}*'.format(id)
    pe_files = glob(file_search)
    for f in pe_files:
        df = pd.read_csv(f, index_col=0)
        types = ['distraction', 'augmentation']
        for t in types:
            if t in plan.keys() and t in df.index:
                if plan[t][0] in df.loc[t].values:
                    return True

    return False


def execute_data_augmentation_plan(data_augmentation_plan_filename, metadata_folder):
    max_num_plans = 4
    with open(data_augmentation_plan_filename, 'rb') as f:
        data_augmentation_plan = pickle.load(f)

    # random.shuffle(data_augmentation_plan)
    # data_augmentation_plan = data_augmentation_plan[:100]

    # i wish to apply plan in sequence for each file, but if the file has multiple plans
    # later plans may get started before eariler one finishes.
    # to workaround this, execute plan[0] for each file, then plan[1] for each file
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for plan_id in range(max_num_plans):
            desc = "Queuing Jobs for id {}".format(plan_id)
            for filename, plan in tqdm(data_augmentation_plan, desc=desc):
                if len(plan) > plan_id:
                    p = plan[plan_id]
                    if check_plan_already_executed(filename, p, metadata_folder):
                        continue
                    if 'augmentation' in p.keys():
                        distr = p['augmentation']
                        jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                                     (filename, filename,),
                                                     dict(augmentation=distr[0], augmentation_param=distr[1])
                                                     )
                                    )
                    elif 'distraction' in p.keys():
                        distr = p['distraction']
                        jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                                     (filename, filename,),
                                                     dict(distraction=distr[0], distraction_param=distr[1])
                                                     ))

        for job in tqdm(jobs, desc="Executing data augmentation plan"):
            r = job.get()
            try:
                df = pd.Series(r).reset_index().set_index('index')
                rfilename = os.path.basename(r['input_file']) + \
                            '_' + str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S")) + '.csv'
                df.to_csv(os.path.join(metadata_folder, rfilename), header=False)
                print(r)
            except Exception:
                print_line()
                print('Got exception')
                print_line()
                print(r)
                print_line()
                print(traceback.print_exc())
                print_line()
                print(sys.exc_info()[0])
                print_line()
            results.append(r)

    return results


def adaptive_video_compress_batch(data_root_dir, data_augmentation_plan_filename=None):
    file_size_map = get_files_size(data_root_dir)
    file_names, file_sizes = list(zip(*file_size_map))
    median_file_size = np.median(file_sizes)
    std = np.std(file_sizes)
    margin = 0.20
    delta = int(median_file_size * margin)
    min_file_size, max_file_size = median_file_size - delta, 3 * std
    num_of_files = len(file_names)

    if data_augmentation_plan_filename is not None:
        with open(data_augmentation_plan_filename, 'rb') as f:
            data_augmentation_plan = pickle.load(f)
        augmented_filenames = list()
        for filename, plan in tqdm(data_augmentation_plan, desc="Build augmented filename list"):
            augmented_filenames.append(os.path.basename(filename))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for idx in tqdm(range(num_of_files), desc="Scheduling compression jobs"):
            # if data_augmentation_plan_filename is supplied then, compress those files only
            if data_augmentation_plan_filename is not None:
                # if current file is not in list of augmented files then dont compress and
                # move on to next file
                if not os.path.basename(file_names[idx]) in augmented_filenames:
                    continue
            jobs.append(pool.apply_async(adaptive_video_compress,
                                         (file_names[idx], min_file_size, max_file_size,)
                                         )
                        )

        for job in tqdm(jobs, desc="Executing adaptive video compression"):
            # print(job.get())
            try:
                results.append(job.get())
            except Exception:
                print_line()
                print(traceback.print_exc())
                print_line()
                print(sys.exc_info()[0])
                print_line()

    df = pd.DataFrame(results).set_index('input_file')
    if not os.path.exists(get_compression_csv_path()):
        df.to_csv(get_compression_csv_path())
    else:
        df.to_csv(get_compression_csv_path(), mode='a', header=False)


def extract_faces_batch(input_videofiles, faces_loc_path, overwrite=False):
    try:
        start_method = torch.multiprocessing.get_start_method()
        torch.multiprocessing.set_start_method('spawn', True)
    except RuntimeError:
        print('Failed to set start method to spawn, CUDA multiprocessing might fail')

    os.makedirs(faces_loc_path, exist_ok=True)
    print(f'Saving json files at: {faces_loc_path}')
    detector = None  # fd.get_face_detector_model()

    num_of_files = len(input_videofiles)
    batch_size = 32

    files_to_process = list()
    if not overwrite:
        for idx in tqdm(range(num_of_files), desc="Checking existing json files"):
            inp = input_videofiles[idx]
            id = os.path.splitext(os.path.basename(inp))[0]
            out_file = os.path.join(faces_loc_path, "{}.json".format(id))
            if os.path.isfile(out_file):
                continue
            else:
                files_to_process.append(inp)
    else:
        files_to_process = input_videofiles

    num_of_files = len(files_to_process)
    # processes = multiprocessing.cpu_count()
    processes = 4
    with multiprocessing.Pool(processes=processes) as pool:
        with tqdm(total=num_of_files) as pbar:
            for v in pool.imap_unordered(
                    partial(fd.extract_faces_from_video, out_dir=faces_loc_path, batch_size=batch_size,
                            detector=detector),
                    files_to_process):
                pbar.update()

    try:
        torch.multiprocessing.set_start_method(start_method)
    except RuntimeError:
        pass


def create_fboxes_video_batch(data_root_dir, faces_loc_path):
    in_videofile = '/home/therock/data2/data_workset/dfdc/train/dfdc_train_part_22/ibhoivgoml.mp4'
    in_videofile = '/home/therock/data2/data_workset/dfdc/train/dfdc_train_part_30/ajxcpxpmof.mp4'
    json_filename = os.path.splitext(os.path.basename(in_videofile))[0] + '.json'
    json_filepath = os.path.join(faces_loc_path, json_filename)
    out_path = get_faces_loc_video_path()
    os.makedirs(out_path, exist_ok=True)
    out_videofile = os.path.join(out_path, os.path.basename(in_videofile))
    fd.recreate_fboxes_video(in_videofile, out_videofile, json_filepath)


def validate_augmented_videos_batch(data_augmentation_plan_filename):
    with open(data_augmentation_plan_filename, 'rb') as f:
        data_augmentation_plan = pickle.load(f)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for filename, plan in tqdm(data_augmentation_plan, desc='Scheduling jobs'):
            jobs.append(pool.apply_async(get_video_integrity, (filename,)))

        for job in tqdm(jobs, desc="Verifying data integrity"):
            r = job.get()
            results.append(r)

    df = pd.DataFrame(results).set_index('filename')
    df.to_csv(get_video_integrity_data_path())


def restore_bad_augmented_files(get_video_integrity_data_path, data_backup_dir, data_root_dir):
    df = pd.read_csv(get_video_integrity_data_path)
    df = df[df['status'] != 'valid']

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        dest = row['filename']
        src = os.path.join(data_backup_dir, os.path.sep.join(dest.split(os.path.sep)[-2:]))
        if os.path.exists(src):
            shutil.copy(src, dest)


def crop_faces_from_videos_batch(input_videofiles, faces_json_path, crop_faces_out_dir):
    print(f'faces_json_path = {faces_json_path}')
    print(f'crop_faces_out_dir = {crop_faces_out_dir}')
    os.makedirs(crop_faces_out_dir, exist_ok=True)

    # multiprocessing.cpu_count()
    with multiprocessing.Pool(2) as pool:
        jobs = []
        results = []
        for filename in tqdm(input_videofiles, desc='Scheduling jobs'):
            jobs.append(pool.apply_async(fd.crop_faces_from_video, (filename, faces_json_path, crop_faces_out_dir,)))

        for job in tqdm(jobs, desc="Exacting faces from videos"):
            r = job.get()
            results.append(r)


def validate_data_loaders(data_root_dir):
    mode = 'train'
    if mode == 'train':
        data = get_all_training_video_filepaths(data_root_dir)

    if mode == 'test':
        data = get_all_training_video_filepaths(data_root_dir)

    if mode == 'valid':
        train_data = get_all_training_video_filepaths(data_root_dir)

    data = data[0:4]
    dataset = DFDCDataset(data, mode=mode)

    batch_size = 2
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                             collate_fn=lambda x: tuple(zip(*x)))
    print(f'len train_data_loader = {len(data_loader)}')

    for idx, sample in enumerate(data_loader):
        video_id, frames, label = sample[0], sample[1], sample[2]
        print(f'---- index = {idx} video_id = {video_id}, frames = {frames}, label = {label} -----')


def main():
    if args.apply_aug_to_sample:
        print('Applying augmentation and distraction to sample file')
        sample_test_file = os.path.join('dfdc_train_part_30', 'ajxcpxpmof.mp4')
        sample_test_file = os.path.join(args.train_data_root_dir, sample_test_file)
        print(f'sample file: {sample_test_file}')

        aug_output_folder = os.path.join(args.train_data_root_dir, 'test_augmentation')
        face_locate_out_folder = os.path.join(args.train_data_root_dir, 'test_augmentation_locate')

        test_data_augmentation(sample_test_file, aug_output_folder)
        test_data_distraction(sample_test_file, aug_output_folder)
        locate_faces_batch(aug_output_folder, face_locate_out_folder)
        print(f'generating poster for samples')
        generate_poster(sample_test_file, aug_output_folder)

    if args.gen_aug_plan:
        print('Generating augmentation plan')
        generate_data_augmentation_plan(args.train_data_root_dir, get_data_aug_plan_pkl_filename(),
                                        get_data_aug_plan_txt_filename())

    if args.apply_aug_to_all:
        print('Executing augmentation plan')
        execute_data_augmentation_plan(get_data_aug_plan_pkl_filename(), get_aug_metadata_path())

    if args.restore_aug_files:
        print('Restoring all augmented files from backup')
        restore_augmented_files(get_aug_metadata_path(), args.train_data_backup_dir, args.train_data_root_dir)

    if args.restore_bad_aug_files:
        print('Restoring invalid augmented files from backup')
        restore_bad_augmented_files(get_video_integrity_data_path(), args.train_data_backup_dir,
                                    args.train_data_root_dir)

    if args.compress_videos:
        print('Compressing videos')
        adaptive_video_compress_batch(args.train_data_root_dir, get_data_aug_plan_pkl_filename())

    if args.extract_faces:
        print('Detecting faces in training videos and saving in json files')
        input_videofiles = get_all_training_video_filepaths(args.train_data_root_dir)
        extract_faces_batch(input_videofiles, get_train_json_faces_data_path())
        print('Detecting faces in validation videos and saving in json files')
        input_videofiles = get_all_validation_video_filepaths(get_validation_data_path())
        extract_faces_batch(input_videofiles, get_valid_json_faces_data_path())
        print('Detecting faces in test videos and saving in json files')
        input_videofiles = get_all_test_video_filepaths(get_test_data_path())
        extract_faces_batch(input_videofiles, get_test_json_faces_data_path())

    if args.create_fboxes_video:
        print('Creating boxes around faces in sample training videos using json files')
        create_fboxes_video_batch(args.train_data_root_dir, get_train_json_faces_data_path())

    if args.validate_aug_video:
        print('Validating augmented files using ffmpeg')
        validate_augmented_videos_batch(get_data_aug_plan_pkl_filename())

    if args.crop_faces:
        print('Cropping faces from training videos')
        input_videofiles = get_all_training_video_filepaths(args.train_data_root_dir)

        crop_faces_from_videos_batch(input_videofiles, get_train_json_faces_data_path(),
                                     get_train_crop_faces_data_path())
        print('Cropping faces from validation videos')
        input_videofiles = get_all_validation_video_filepaths(get_validation_data_path())
        crop_faces_from_videos_batch(input_videofiles, get_valid_json_faces_data_path(),
                                     get_valid_crop_faces_data_path())
        print('Cropping faces from test videos')
        input_videofiles = get_all_test_video_filepaths(get_test_data_path())

        crop_faces_from_videos_batch(input_videofiles, get_test_json_faces_data_path(),
                                     get_test_crop_faces_data_path())

    if args.consol_train_labels:
        print('Consolidate all filenames and related labels for training data')
        csv_file = get_train_labels_csv_filepath()
        consolidate_training_labels(args.train_data_root_dir, csv_file)
        print(f'Saved {csv_file}')

    if args.validate_data_loaders:
        print('Validating sample dataloaders')
        validate_data_loaders(args.train_data_root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data pre-processing for DFDC')

    parser.add_argument('--apply_aug_to_sample', action='store_true',
                        help='Apply augmentation and distractions to a file',
                        default=False)
    parser.add_argument('--gen_aug_plan', action='store_true',
                        help='Generate augmentation and distractions plan',
                        default=False)
    parser.add_argument('--apply_aug_to_all', action='store_true',
                        help='Apply augmentation and distractions to all samples, per plan',
                        default=False)
    parser.add_argument('--train_data_root_dir', help='Root dir for DFDC train dataset',
                        default=get_train_data_path())
    parser.add_argument('--train_data_backup_dir', help='Root dir for backup DFDC train dataset',
                        default=get_backup_train_data_path())
    parser.add_argument('--restore_aug_files', action='store_true',
                        help='Restore all augmented files',
                        default=False)
    parser.add_argument('--restore_bad_aug_files', action='store_true',
                        help='Restore bad augmented files',
                        default=False)
    parser.add_argument('--compress_videos', action='store_true',
                        help='Compress all videos (adaptive compression to maintain approx median filesize)',
                        default=False)
    parser.add_argument('--extract_faces', action='store_true',
                        help='Detect faces from videos and store json for each face location',
                        default=False)
    parser.add_argument('--create_fboxes_video', action='store_true',
                        help='Use json file to overlay box on faces',
                        default=False)
    parser.add_argument('--validate_aug_video', action='store_true',
                        help='Use ffmpeg to check if augmented videos are valid',
                        default=False)
    parser.add_argument('--crop_faces', action='store_true',
                        help='Crop faces and generate image from videos, using json files',
                        default=False)
    parser.add_argument('--consol_train_labels', action='store_true',
                        help='Consolidate all training samples with their labels',
                        default=False)
    parser.add_argument('--validate_data_loaders', action='store_true',
                        help='Create sample data loaders and validate its functionality',
                        default=False)

    args = parser.parse_args()
    print(args)
    create_assets_placeholder()

    main()
