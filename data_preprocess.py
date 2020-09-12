import multiprocessing
from functools import partial
from tqdm import tqdm
import data_utils.augmentation as augmentation
import data_utils.distractions as distractions
import os
import cv2
from utils import *
from glob import glob
from data_utils.utils import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle


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


def locate_faces(input_root, output_root):
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
    pprint(distraction_plan)

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


def generate_poster(org_file, aug_output_folder, frames_count=10, out_res=None):
    id = os.path.splitext(os.path.basename(org_file))[0]
    org_out_images = os.path.join(aug_output_folder, id + '_original')
    # extract_images_from_video(org_file, org_out_images)
    sub_fol = []
    for f in glob(aug_output_folder + '/*'):
        if os.path.isdir(f):
            sub_fol.append(f)
    sub_fol_count = len(sub_fol)
    print(f'sub_fol_count {sub_fol_count}')
    result_grid_filename = os.path.join(aug_output_folder, 'poster.jpg')
    result_figsize_resolution = 40  # 1 = 100px

    total_frames = 300
    step = int(total_frames / frames_count)
    images_list = ['{}.jpg'.format(i) for i in range(0, total_frames, step)]
    images_count = len(images_list)
    print('Images: ', images_list)
    print('Images count: ', images_count)

    out_res_per_image = 1920, 1080
    result_figsize_resolution = out_res_per_image[0] * images_count, out_res_per_image[1] * images_count
    print(f'result_figsize_resolution = {result_figsize_resolution}')
    fig, axes = plt.subplots(sub_fol_count, frames_count,
                             figsize=(40, 40)
                             )

    fol_n_replace_str = id + '_'
    for i, folder in enumerate(sub_fol):
        fol_n = folder.replace(fol_n_replace_str, '')
        axes[i, 0].set_ylabel(fol_n)
        for j, frame in enumerate(images_list):
            plt_image = plt.imread(folder + '/' + frame)
            axes[i, j].imshow(plt_image)
            axes[i, j].axis('off')
            """
            if j == 0:
                n = os.path.basename(folder)
                # axes[i, j].set_title(n, loc='left',y=True)
                axes[i, j].set_ylabel(n)
            else:
                axes[i, j].set_title('')
            """
            if i == 0:
                fnum = frame.replace('.jpg', '')
                axes[0, j].set_title('frame {}'.format(fnum))
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            # print(folder + '/' + frame)
    plt.savefig(result_grid_filename)

    print(f'generated {result_grid_filename}')




def main():
    input_file = '/home/therock/dfdc/train/dfdc_train_part_30/ajxcpxpmof.mp4'
    data_root_dir = '/home/therock/dfdc/train/'
    aug_output_folder = '/home/therock/dfdc/test_augmentation_2/'
    output_track_folder = os.path.join(aug_output_folder, 'tracked')
    generate_poster(input_file, aug_output_folder)


if __name__ == '__main__':
    main()
