import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
from data_utils.utils import *
from utils import *
from PIL import Image


class DFDCDataset(Dataset):
    def __init__(self, data, mode=None, transform=None, max_num_frames=10, frame_dim=256, device=None):
        super().__init__()
        self.data = data
        self.mode = mode
        if self.mode == 'train':
            self.crops_dir = get_train_crop_faces_data_path()
            self.labels_csv = get_train_labels_csv_filepath()
        elif self.mode == 'valid':
            self.crops_dir = get_valid_crop_faces_data_path()
            self.labels_csv = get_valid_labels_csv_filepath()
        elif self.mode == 'test':
            self.crops_dir = get_test_crop_faces_data_path()
            self.labels_csv = get_test_labels_csv_filepath()
        else:
            raise Exception("Invalid mode in DFDCDataset passed")

        self.lookup_table = self._generate_loopup_table()
        self.data_len = len(self.data)
        self.transform = transform
        self.max_num_frames = max_num_frames
        self.frame_dim = frame_dim

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        video_filepath = self.data[index]
        video_filename = os.path.basename(video_filepath)
        video_id = os.path.splitext(video_filename)[0]
        crops_id_path = os.path.join(self.crops_dir, video_id)
        debug = False
        if not debug:
            frame_names = glob(crops_id_path + '/*_0.png')
            num_of_frames = len(frame_names)
            if num_of_frames == 0:
                return None
            frame_names = sorted(frame_names, key=alpha_sort_keys)
            if num_of_frames > self.max_num_frames:
                frame_names = frame_names[0:self.max_num_frames]
            frames = list()
            for f in frame_names:
                image = Image.open(f)
                if self.transform is not None:
                    image = self.transform(image)
                    # print(f.shape)
                frames.append(image)

            if num_of_frames < self.max_num_frames:
                delta = self.max_num_frames - num_of_frames
                for f in range(delta):
                    frames.append(torch.zeros_like(frames[0]))
        else:
            frames = list()
            for f in range(self.max_num_frames):
                frames.append(torch.ones(3, self.frame_dim, self.frame_dim) * (random.randint(1, 5)))

        frames = torch.stack(frames, dim=0)
        label = torch.tensor(self.lookup_table[video_filename], dtype=torch.long)
        item = (video_id, frames, label)
        return item

    def _generate_loopup_table(self):
        df = pd.read_csv(self.labels_csv, squeeze=True, index_col=0)
        return df.to_dict()
