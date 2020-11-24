import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
from data_utils.utils import *
from utils import *
from PIL import Image


class DFDCDataset(Dataset):
    def __init__(self, data, mode=None, transform=None, max_num_frames=10, frame_dim=256, random_sorted=False,
                 device=None, fill_empty=True, label_smoothing=0):
        super().__init__()
        self.data = data
        self.mode = mode
        self.label_smoothing = 0  # use only in training, so update to param passed in train mode
        if self.mode == 'train':
            self.crops_dir = get_train_crop_faces_data_path()
            self.labels_csv = get_train_labels_csv_filepath()
            self.label_smoothing = label_smoothing
        elif self.mode == 'valid':
            self.crops_dir = get_valid_crop_faces_data_path()
            self.labels_csv = get_valid_labels_csv_filepath()
        elif self.mode == 'test':
            self.crops_dir = get_test_crop_faces_data_path()
            self.labels_csv = get_test_labels_csv_filepath()
        else:
            raise Exception("Invalid mode in DFDCDataset passed")

        self.lookup_table = self._generate_lookup_table()
        self.data_len = len(self.data)
        self.transform = transform
        self.max_num_frames = max_num_frames
        self.frame_dim = frame_dim
        self.random_sorted = random_sorted
        self.fill_empty = fill_empty

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        video_filepath = self.data[index]
        video_filename = os.path.basename(video_filepath)
        video_id = os.path.splitext(video_filename)[0]
        crops_id_path = os.path.join(self.crops_dir, video_id)
        debug = False
        if not debug:
            frame_names = glob(crops_id_path + '/*_*.png')
            # find number of faces detected in this set of crops
            face_ids = list(
                set([i.replace(crops_id_path + '/', '').replace('.png', '').split('_')[1] for i in frame_names]))
            if len(face_ids) > 1:
                # if number of faces were more than one then chose a face randomly, and get all crops for that face.
                face_id_rand = random.choice(face_ids)
                frame_names = glob(crops_id_path + '/*_{}.png'.format(face_id_rand))
            num_of_frames = len(frame_names)
            if num_of_frames == 0:
                return None
            if self.random_sorted:
                # select max_num_frames frames randomly then sort them
                frame_names = random.sample(frame_names, min(self.max_num_frames, num_of_frames))

            frame_names = sorted(frame_names, key=alpha_sort_keys)
            if num_of_frames > self.max_num_frames:
                frame_names = frame_names[0:self.max_num_frames]
            frames = list()
            for f in frame_names:
                image = Image.open(f)
                if self.transform is not None:
                    image = self.transform(image)
                frames.append(image)

            if self.fill_empty:
                if num_of_frames < self.max_num_frames:
                    delta = self.max_num_frames - num_of_frames
                    for f in range(delta):
                        frames.append(torch.zeros_like(frames[0]))
        else:
            frames = list()
            for f in range(self.max_num_frames):
                frames.append(torch.ones(3, self.frame_dim, self.frame_dim) * (random.randint(1, 5)))

        frames = torch.stack(frames, dim=0)
        label = self.lookup_table[video_filename]
        if self.label_smoothing != 0:
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
        label = torch.tensor(label)
        item = (video_id, frames, label)
        return item

    def _generate_lookup_table(self):
        df = pd.read_csv(self.labels_csv, squeeze=True, index_col=0)
        return df.to_dict()


class DFDCDatasetSimple(Dataset):
    def __init__(self, mode=None, transform=None, data_size=1, dataset=None, label_smoothing=0):
        super().__init__()
        self.mode = mode
        self.label_smoothing = 0  # use only in training, so update to param passed in train mode
        if mode == 'train':
            if dataset == 'plain':
                self.labels_csv = get_train_frame_label_csv_path()
                self.crops_dir = get_train_crop_faces_data_path()
            elif dataset == 'optical':
                self.labels_csv = get_train_optframe_label_csv_path()
                self.crops_dir = get_train_optical_png_data_path()
            elif dataset == 'mri':
                self.labels_csv = get_train_mriframe_label_csv_path()
                self.crops_dir = get_train_mrip2p_png_data_path()
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')

            self.label_smoothing = label_smoothing
        elif mode == 'valid':
            if dataset == 'plain':
                self.labels_csv = get_valid_frame_label_csv_path()
                self.crops_dir = get_valid_crop_faces_data_path()
            elif dataset == 'optical':
                self.labels_csv = get_valid_optframe_label_csv_path()
                self.crops_dir = get_valid_optical_png_data_path()
            elif dataset == 'mri':
                self.labels_csv = get_valid_mriframe_label_csv_path()
                self.crops_dir = get_valid_mrip2p_png_data_path()
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')

        elif mode == 'test':
            if dataset == 'plain':
                self.labels_csv = get_test_frame_label_csv_path()
                self.crops_dir = get_test_crop_faces_data_path()
            elif dataset == 'optical':
                self.labels_csv = get_test_optframe_label_csv_path()
                self.crops_dir = get_test_optical_png_data_path()
            elif dataset == 'mri':
                self.labels_csv = get_test_mriframe_label_csv_path()
                self.crops_dir = get_test_mrip2p_png_data_path()
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')
        else:
            raise Exception('Bad mode in DFDCDatasetSimple')

        self.data_df = pd.read_csv(self.labels_csv)
        if data_size < 1:
            total_data_len = int(len(self.data_df) * data_size)
            self.data_df = self.data_df.iloc[0:total_data_len]
        self.data_dict = self.data_df.to_dict(orient='records')
        self.data_len = len(self.data_df)
        self.transform = transform

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        while True:
            try:
                item = self.data_dict[index].copy()
                frame = Image.open(os.path.join(self.crops_dir, str(item['video_id']), item['frame']))
                if self.transform is not None:
                    frame = self.transform(frame)
                item['frame_tensor'] = frame
                if self.label_smoothing != 0:
                    label = np.clip(item['label'], self.label_smoothing, 1 - self.label_smoothing)
                else:
                    label = item['label']
                item['label'] = torch.tensor(label)
                return item
            except Exception:
                #print(f"bad {os.path.join(self.crops_dir, str(item['video_id']), item['frame'])}")
                index = random.randint(0, self.data_len)


class MRIDataset(Dataset):
    def __init__(self, transforms=None, mode="train"):
        self.transforms = transforms

        if mode == "train":
            self.data_csv = get_xray_pairs_train_csv()
        elif mode == "test":
            self.data_csv = get_xray_pairs_test_csv()
        else:
            raise Exception("Unknown mode")

        self.df = pd.read_csv(self.data_csv)
        self.data_dict = self.df.to_dict(orient='records')
        self.df_len = len(self.df)

    def __getitem__(self, index):
        while True:
            try:
                item = self.data_dict[index].copy()
                img_A = Image.open(str(item['face_image']))
                img_B = Image.open(str(item['xray_image']))

                if np.random.random() < 0.5:
                    img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                    img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

                if self.transforms:
                    img_A = self.transforms(img_A)
                    img_B = self.transforms(img_B)

                return {"A": img_A, "B": img_B}

            except Exception:
                index = random.randint(0, self.df_len)

    def __len__(self) -> int:
        return self.df_len


class SimpleImageFolder(Dataset):
    def __init__(self, root, transforms_=None):
        self.root = root
        all_files = glob(root + "/*.png")
        self.data_list = [os.path.abspath(f) for f in all_files]
        self.data_len = len(self.data_list)
        self.transforms = transforms_

    def __getitem__(self, index):
        img_name = self.data_list[index]
        img = Image.open(self.data_list[index])
        if self.transforms:
            img = self.transforms(img)
        return img, img_name

    def __len__(self) -> int:
        return self.data_len
