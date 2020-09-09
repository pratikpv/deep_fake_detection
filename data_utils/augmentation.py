from functools import partial
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
from PIL import Image
import os
import numpy as np
from skimage.util import random_noise
import time
import torch

"""
def create_video_from_images_(images, output_video_filename, fps=30, res=(1920, 1080)):
    video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, res)
    for image in images:
        print(image)
        image = cv2.imread(image)
        image = cv2.resize(image, res)
        video.write(image)

    # cv2.destroyAllWindows()
    video.release()
    print(f'generated {output_video_filename}')
"""


def create_video_from_images(images, output_video_filename, fps=30, res=(1920, 1080)):
    video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, res)
    for image in images:
        video.write(image)
    video.release()


def apply_blur_to_image(image, augmentation_param):
    return cv2.blur(image, augmentation_param['ksize'])


def apply_noise_to_image(image=None, augmentation_param=None, mode=None):
    image = random_noise(image, mode=mode)
    image = np.array(255 * image, dtype=np.uint8)
    return image


augmentation_mapping = {
    'blur': apply_blur_to_image,
    'gaussian': partial(apply_noise_to_image, mode='gaussian'),
    'speckle': partial(apply_noise_to_image, mode='speckle'),
    's&p': partial(apply_noise_to_image, mode='s&p'),
    'pepper': partial(apply_noise_to_image, mode='pepper'),
    'salt': partial(apply_noise_to_image, mode='salt'),
    'poisson': partial(apply_noise_to_image, mode='poisson'),
    'localvar': partial(apply_noise_to_image, mode='localvar')
}


def apply_augmentation_to_videofile(input_video_filename, output_video_filename, augmentation=None,
                                    augmentation_param=None, res=None, save_intermdt_files=False):
    capture = cv2.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    if augmentation in augmentation_mapping.keys():
        augmentation_func = augmentation_mapping[augmentation]
    else:
        raise Exception("Unknown augmentation supplied")

    out_images_path = os.path.join(os.path.dirname(output_video_filename),
                                   os.path.splitext(os.path.basename(output_video_filename))[0],
                                   )
    if save_intermdt_files:
        os.makedirs(out_images_path, exist_ok=True)

    frames = list()
    for i in range(frames_num):
        # t = time.time()
        capture.grab()
        success, frame = capture.retrieve()
        # if i % 20 !=0:
        #    continue
        if not success:
            continue
        if res is not None:
            frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)

        frame = augmentation_func(image=frame, augmentation_param=augmentation_param)
        if save_intermdt_files:
            out_image_name = os.path.join(out_images_path, "{}.jpg".format(i))
            # print(f'saving {out_image_name}')
            cv2.imwrite(out_image_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # print('Done in', (time.time() - t))

        frames.append(frame)
    out_res = org_res
    if res is not None:
        out_res = res
    create_video_from_images(frames, output_video_filename, fps=org_fps, res=out_res)

    return


def locate_face_in_videofile(input_filepath=None, outfile_filepath=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)

    capture = cv2.VideoCapture(input_filepath)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        face_box = list(detector.detect(frame, landmarks=False))[0]
        if face_box is not None:
            for f in range(len(face_box)):
                fc = list(face_box[f])
                cv2.rectangle(frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
        frames.append(frame)
    create_video_from_images(frames, outfile_filepath, fps=org_fps, res=org_res)

