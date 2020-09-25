import torch
from features.encoders import DeepFakeEncoder
from glob import glob
from PIL import Image
from torchvision.transforms import transforms
import os

imsize = 224
loader = transforms.Compose([transforms.ToTensor()])


def generate_cnn_video_encodings(item_dir_path=None, features_dir=None, overwrite=False):
    item_name = os.path.basename(item_dir_path)
    feature_item_dir = os.path.join(features_dir, item_name)
    os.makedirs(feature_item_dir, exist_ok=True)
    print(feature_item_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    encoder = DeepFakeEncoder().to(device)
    for imf in glob(item_dir_path + "/*"):
        f_id = os.path.splitext(os.path.basename(imf))[0]
        ff = os.path.join(feature_item_dir, f_id + '.vec')
        if not overwrite and os.path.isfile(ff):
            continue
        image = Image.open(imf)
        image = loader(image).float()
        image = image.unsqueeze(0)
        print(f'shape = {image.shape}')
        image = image.to(device)
        features = encoder(image)
        torch.save(features, ff)
        print(f'features = {features.shape}')


