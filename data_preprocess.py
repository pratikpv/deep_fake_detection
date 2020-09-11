import multiprocessing
from functools import partial
from tqdm import tqdm
import data_utils.augmentation as augmentation
import data_utils.distractions as distractions
import os
import cv2
from utils import *
import glob


def data_augmentation():
    inp = '/home/therock/dfdc/train/dfdc_train_part_19/jexvqmufit.mp4'
    outp_root = '/home/therock/dfdc/test_augmentation/'
    os.makedirs(outp_root, exist_ok=True)

    methods = [
        # ('blur', {'ksize': (10, 10)}),
        # ('gaussian', None),
        # ('speckle', None),
        # ('s&p', None),
        # ('pepper', None),
        # ('salt', None),
        # ('poisson', None),
        # ('localvar', None)
        ('contrast', None)
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


def data_distraction():
    inp = '/home/therock/dfdc/train/dfdc_train_part_19/jexvqmufit.mp4'
    outp_root = '/home/therock/dfdc/test_distraction/'
    os.makedirs(outp_root, exist_ok=True)

    """
    Sample paramas. Use '$RANDOM$' as value for any key to select random value
    
    static_text_param = {
        'text': 'pratik',
        'loc': (10, 250),
        'color': (255, 0, 0),
        'fontScale': 3,
        'thickness': 3,
    }
    
    text_dir: l_to_r, r_to_l, t_to_b, b_to_t
    rolling_text_param = {
        'text': '$RANDOM$',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'fontScale': '$RANDOM$',
        'thickness': '$RANDOM$',
        'text_dir': 'l_to_r'
    }
    
    """

    static_text_param = {
        'text': '$RANDOM$',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'fontScale': '$RANDOM$',
        'thickness': '$RANDOM$'
    }
    spontaneous_text_param = {
        'text': '$RANDOM$',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'fontScale': '$RANDOM$',
        'thickness': '$RANDOM$',
        'rate': 0.1
    }

    spontaneous_shape_param = {
        'shape': 'rectangle',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'size': 'large',
        'rate': 0.1
    }

    rolling_text_param = {
        'text': '$RANDOM$',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'fontScale': '$RANDOM$',
        'thickness': '$RANDOM$',
        'rolling_dir': 'b_to_t'
    }

    static_shape_param = {
        'shape': 'rectangle',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'size': 'large'
    }

    rolling_shape_param = {
        'shape': 'circle',
        'loc': '$RANDOM$',
        'color': '$RANDOM$',
        'size': 'large',
        'rolling_dir': 'b_to_t'
    }

    methods = [
        # ('static_text', static_text_param),
        # ('rolling_text', rolling_text_param),
        # ('static_shape', static_shape_param),
        # ('rolling_shape', rolling_shape_param),
        # ('spontaneous_text', spontaneous_text_param),
        ('spontaneous_shape', spontaneous_shape_param)
    ]

    out_id = os.path.splitext(os.path.basename(inp))[0]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for distract in methods:
            distract_func, distract_param = distract
            outfile = os.path.join(outp_root, out_id + '_' + distract_func + '.mp4')
            jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                     (inp, outfile,),
                                     dict(distraction=distract_func, distraction_param=distract_param)
                                     ))

        for job in tqdm(jobs, desc="Applying augmentation"):
            results.append(job.get())


def main():
    # data_augmentation()
    # locate_faces()
    data_distraction()


if __name__ == '__main__':
    main()
