import torch
from features.encoders import *
from glob import glob
from PIL import Image
from torchvision.transforms import transforms
import os
from utils import *


def get_simple_transforms(imsize):
    return transforms.Compose([transforms.Resize((imsize, imsize)),
                               transforms.ToTensor()]
                              )


def generate_cnn_video_encodings(item_dir_path=None, features_dir=None, overwrite=False):
    item_name = os.path.basename(item_dir_path)
    feature_item_dir = os.path.join(features_dir, item_name)
    os.makedirs(feature_item_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    encoder_name = get_default_cnn_encoder_name()
    encoder = DeepFakeEncoder(encoder_name).to(device)
    imsize = encoder_params[encoder_name]["imsize"]
    trans = get_simple_transforms(imsize)
    for imf in glob(item_dir_path + "/*"):
        f_id = os.path.splitext(os.path.basename(imf))[0]
        ff = os.path.join(feature_item_dir, f_id + '.vec')
        if not overwrite and os.path.isfile(ff):
            continue
        image = Image.open(imf)
        image = trans(image).float()
        image = image.unsqueeze(0)
        # print(f'shape = {image.shape}')
        image = image.to(device)
        features = encoder(image)
        torch.save(features, ff)
        # print(f'features = {features.shape}')


def generate_cnn_video_encodings_batch(item_dir_path=None, features_dir=None, batch_size=32, overwrite=False):
    item_name = os.path.basename(item_dir_path)
    feature_item_dir = os.path.join(features_dir, item_name)
    os.makedirs(feature_item_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    encoder_name = get_default_cnn_encoder_name()
    encoder = DeepFakeEncoder(encoder_name).to(device)
    imsize = encoder_params[encoder_name]["imsize"]
    trans = get_simple_transforms(imsize)

    batches = list()
    imf_list = list()
    images_list = list()
    ff_list = list()

    for imf in glob(item_dir_path + "/*"):
        f_id = os.path.splitext(os.path.basename(imf))[0]
        ff = os.path.join(feature_item_dir, f_id + '.vec')
        if not overwrite and os.path.isfile(ff):
            continue
        imf_list.append(imf)
        image = Image.open(imf)
        image = trans(image).float()
        image = image.to(device)
        images_list.append(image)
        ff_list.append(ff)

    num_files = len(imf_list)
    for i in range(0, num_files, batch_size):
        end = i + batch_size
        if end > num_files:
            end = num_files
        batches.append((imf_list[i:end], ff_list[i:end], images_list[i:end]))

    for j, frames_list in enumerate(batches):
        imf_b, ff_b, images_b = frames_list
        frame_items = torch.stack(images_b)
        features = encoder(frame_items)
        b = features.shape[0]
        for f in range(b):
            # print(f'saving features[{f}] to {ff_b[f]}')
            torch.save(features[f], ff_b[f])


def verify_cnn_video_encodings(item_dir_path=None, features_dir=None, overwrite=True):
    item_name = os.path.basename(item_dir_path)
    feature_item_dir = os.path.join(features_dir, item_name)
    os.makedirs(feature_item_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    encoder_name = get_default_cnn_encoder_name()
    encoder = DeepFakeEncoder(encoder_name).to(device)
    imsize = encoder_params[encoder_name]["imsize"]
    trans = get_simple_transforms(imsize)
    for imf in glob(item_dir_path + "/*"):
        f_id = os.path.splitext(os.path.basename(imf))[0]
        ff = os.path.join(feature_item_dir, f_id + '.vec')
        image = Image.open(imf)
        image = trans(image).float()
        image = image.unsqueeze(0)
        image = image.to(device)
        features = encoder(image)
        print(imf)
        print(ff)
        features = features.to('cpu')
        features_load = torch.load(ff)
        features_load = features_load.to('cpu')
        if features.equal(features_load):
            print('match')
        else:
            print('not-match')
