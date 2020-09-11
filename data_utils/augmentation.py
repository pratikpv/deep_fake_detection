from functools import partial
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
from PIL import Image
import os
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import time
import torch
from data_utils.utils import *
import random


def apply_blur_to_image(image, augmentation_param):
    return cv2.blur(image, augmentation_param['ksize'])


def apply_noise_to_image(image=None, augmentation_param=None, mode=None):
    image = random_noise(image, mode=mode)
    image = np.array(255 * image, dtype=np.uint8)
    return image


def apply_contrast_to_image(image, augmentation_param):
    contrast = augmentation_param['contrast_value']
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image


def apply_brightness_to_image(image, augmentation_param):
    brightness = augmentation_param['brightness_value']
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    return image


def apply_graysclae_to_image(image, augmentation_param):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_supported_augmentation_methods():
    return augmentation_mapping.keys()


def get_random_blur_value():
    val = random.randint(7, 12)
    return val, val


def get_random_contrast_value():
    return random.randint(-20, 20)


def get_random_brightness_value():
    return random.randint(-20, 20)


def get_random_angle_value():
    return random.randint(-20, 20)


def get_supported_res_value():
    return [(800, 600), (1280, 720)]


def get_random_res_value():
    return random.choice(get_supported_res_value())


def apply_rotation_to_image(image, augmentation_param):
    angle = augmentation_param['angle']
    row, col, c = np.asarray(image).shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def apply_flip_horizontal(image, augmentation_param):
    return cv2.flip(image, 1)


def apply_rescale_to_image(image, augmentation_param):
    res = augmentation_param['res']
    return cv2.resize(image, res, interpolation=cv2.INTER_AREA)


augmentation_mapping = {
    'blur': apply_blur_to_image,
    'gaussian': partial(apply_noise_to_image, mode='gaussian'),
    'speckle': partial(apply_noise_to_image, mode='speckle'),
    's&p': partial(apply_noise_to_image, mode='s&p'),
    'pepper': partial(apply_noise_to_image, mode='pepper'),
    'salt': partial(apply_noise_to_image, mode='salt'),
    'poisson': partial(apply_noise_to_image, mode='poisson'),
    'localvar': partial(apply_noise_to_image, mode='localvar'),
    'contrast': apply_contrast_to_image,
    'brightness': apply_brightness_to_image,
    'rotation': apply_rotation_to_image,
    'flip_horizontal': apply_flip_horizontal,
    'rescale': apply_rescale_to_image,
}


def prepare_augmentation_param(augmentation, augmentation_param, frame_num, res):
    if augmentation == 'blur':
        if frame_num == 0:
            if augmentation_param['ksize'] == random_setting:
                augmentation_param['ksize'] = get_random_blur_value()

    if augmentation == 'contrast':
        if frame_num == 0:
            if augmentation_param['contrast_value'] == random_setting:
                augmentation_param['contrast_value'] = get_random_contrast_value()

    if augmentation == 'brightness':
        if frame_num == 0:
            if augmentation_param['brightness_value'] == random_setting:
                augmentation_param['brightness_value'] = get_random_contrast_value()

    if augmentation == 'rotation':
        if frame_num == 0:
            if augmentation_param['angle'] == random_setting:
                augmentation_param['angle'] = get_random_angle_value()

    if augmentation == 'rescale':
        if frame_num == 0:
            if augmentation_param['res'] == random_setting:
                augmentation_param['res'] = get_random_res_value()

    return augmentation_param


def apply_augmentation_to_videofile(input_video_filename, output_video_filename, augmentation=None,
                                    augmentation_param=None, save_intermdt_files=False):
    # t = time.time()
    if augmentation in get_supported_augmentation_methods():
        augmentation_func = augmentation_mapping[augmentation]
    else:
        raise Exception("Unknown augmentation supplied")

    capture = cv2.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    if augmentation_param is None:
        augmentation_param = dict()
    augmentation_param['image_width'] = res[0]
    augmentation_param['image_height'] = res[1]

    out_images_path = os.path.join(os.path.dirname(output_video_filename),
                                   os.path.splitext(os.path.basename(output_video_filename))[0],
                                   )
    if save_intermdt_files:
        os.makedirs(out_images_path, exist_ok=True)

    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        augmentation_param = prepare_augmentation_param(augmentation, augmentation_param, i, res)
        if augmentation == 'rescale':
            res = augmentation_param['res']
        frame = augmentation_func(image=frame, augmentation_param=augmentation_param)

        if save_intermdt_files:
            out_image_name = os.path.join(out_images_path, "{}.jpg".format(i))
            print(f'saving {out_image_name}')
            cv2.imwrite(out_image_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        frames.append(frame)

    create_video_from_images(frames, output_video_filename, fps=org_fps, res=res)
    # print('Done in', (time.time() - t))
    # print(output_video_filename)
    return
