import multiprocessing
from functools import partial
from tqdm import tqdm
import data_utils.augmentation as augmentation
import os
import cv2
from utils import *
import glob

def data_augmentation():
    inp = '/home/therock/dfdc/train/dfdc_train_part_19/jexvqmufit.mp4'
    outp_root = '/home/therock/dfdc/test_augmentation/'
    os.makedirs(outp_root, exist_ok=True)

    methods = [
        ('blur', {'ksize': (10, 10)}),
        ('gaussian', None),
        ('speckle', None),
        ('s&p', None),
        ('pepper', None),
        ('salt', None),
        ('poisson', None),
        ('localvar', None)
    ]

    out_id = os.path.splitext(os.path.basename(inp))[0]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for aug in methods:
            aug_func, aug_param = aug
            outfile = os.path.join(outp_root, out_id + '_' + aug_func + '.mp4')
            jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                         (inp, outfile,),
                                         dict(augmentation_param=aug_param, augmentation=aug_func)
                                         )
                        )

        for job in tqdm(jobs, desc="Applying augmentation"):
            results.append(job.get())

def locate_faces():
    input_root = '/home/therock/dfdc/test_augmentation/'
    output_root = os.path.join(input_root, 'tracked')
    os.makedirs(output_root, exist_ok=True)
    input_filepath_list = glob.glob(input_root + '/*.mp4')
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


def main():
    # data_augmentation()
    locate_faces()


if __name__ == '__main__':
    main()
