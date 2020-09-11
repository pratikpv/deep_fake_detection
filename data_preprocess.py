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


def test_data_augmentation(input_file, output_folder):
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


def main():
    input_file = '/home/therock/dfdc/train/dfdc_train_part_30/ajxcpxpmof.mp4'
    aug_output_folder = '/home/therock/dfdc/test_augmentation/'
    output_track_folder = os.path.join(aug_output_folder, 'tracked')
    test_data_augmentation(input_file, aug_output_folder)
    test_data_distraction(input_file, aug_output_folder)
    locate_faces(aug_output_folder, output_track_folder)


if __name__ == '__main__':
    main()
