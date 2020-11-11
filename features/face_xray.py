import data_utils.face_detection as fd
import numpy as np
import os
from PIL import Image
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
import math
import dlib
from skimage.measure import compare_ssim
from scipy.spatial import ConvexHull
from skimage import measure
import skimage.draw
import random
from glob import glob
from imutils import face_utils
from utils import *
import pandas as pd


def gen_xray(image1_path, image2_path, xray_path, res=(224, 224)):
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    image1 = cv2.resize(image1, res, interpolation=cv2.INTER_AREA)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    image2 = cv2.resize(image2, res, interpolation=cv2.INTER_AREA)

    d, a = compare_ssim(image1, image2, multichannel=True, full=True)
    a = 1 - a
    xray = (a * 255).astype(np.uint8)
    # xray = cv2.cvtColor(xray, cv2.COLOR_BGR2GRAY)
    # xray = cv2.cvtColor(xray, cv2.COLOR_BGR2RGB)
    cv2.imwrite(xray_path, xray)


def gen_xray_per_folder(folder1, folder2, xray_basedir, overwrite=True):
    """

    :param folder1: real images
    :param folder2: fake images
    :param xray_basedir: 
    :param overwrite:
    :return:
    """
    folder1_path = os.path.join(get_train_crop_faces_data_path(), folder1)
    folder2_path = os.path.join(get_train_crop_faces_data_path(), folder2)
    dest_folder = os.path.join(xray_basedir, os.path.basename(folder2_path))
    if not overwrite and os.path.isdir(dest_folder):
        return None
    f1_all_files = glob(folder1_path + "/*")
    os.makedirs(dest_folder, exist_ok=True)
    df = pd.DataFrame(columns=['real_image', 'fake_image', 'xray_image'])
    for f1_file in f1_all_files:
        f1_file_base = os.path.basename(f1_file)
        f2_file = os.path.join(folder2_path, f1_file_base)
        if os.path.isfile(f2_file):
            xray_path = os.path.join(dest_folder, f1_file_base)
            gen_xray(f1_file, f2_file, xray_path)
            item = {'real_image': f1_file,
                    'fake_image': f2_file,
                    'xray_image': xray_path}
            df = df.append(item, ignore_index=True)
    return df
