import torch
from features.encoders import *
from glob import glob
from PIL import Image
from torchvision.transforms import transforms
import os
from utils import *
import features.vector_flow.vector_flow as vf


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


def generate_optical_flow_data(crop_faces_data_path, optical_flow_data_path, optical_png_data_path,
                               video_id, imsize, overwrite=False, delete_flow_dir=True):
    if not overwrite and os.path.isdir(os.path.join(optical_png_data_path, video_id)):
        return

    os.makedirs(os.path.join(optical_flow_data_path, video_id), exist_ok=True)
    os.makedirs(os.path.join(optical_png_data_path, video_id), exist_ok=True)

    flow_model = vf.get_optical_flow_model()
    resize_dim = (imsize, imsize)

    vid_path = os.path.join(crop_faces_data_path, video_id)
    frame_names = glob(vid_path + "/*.png")
    face_ids = list(
        set([i.replace(vid_path + '/', '').replace('.png', '').split('_')[1] for i in frame_names]))

    image_pair = []
    for fid in face_ids:
        frame_names_per_id = glob(vid_path + "/*_{}.png".format(fid))
        frame_names_per_id = sorted(frame_names_per_id, key=alpha_sort_keys)
        flen = len(frame_names_per_id)
        for i in range(0, flen - 1):
            image_pair.append((frame_names_per_id[i], frame_names_per_id[i + 1]))

    for pair in image_pair:
        image1, image2 = pair[0], pair[1]
        i1 = os.path.splitext(os.path.basename(image1))[0]
        i2 = os.path.splitext(os.path.basename(image2))[0]
        flow_file = i1 + '_to_' + i2 + '.flo'
        flow_png = i1 + '_to_' + i2 + '.png'
        flow_file_path = os.path.join(optical_flow_data_path, video_id, flow_file)
        flow_image_path = os.path.join(optical_png_data_path, video_id, flow_png)
        # print('*' * 50)
        # print(f'image1 {image1} -> image2 {image2}')
        # print(f'flow_file_path = {flow_file_path}')
        # print(f'flow_image_path = {flow_image_path}')
        vf.gen_vector_flow_png(flow_model, image1, image2, flow_file_path, flow_image_path, resize_dim)

    if delete_flow_dir:
        shutil.rmtree(os.path.join(optical_flow_data_path, video_id))
