import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
from data_utils.utils import *
from utils import *


class DFDCDataset(Dataset):
    def __init__(self, data, mode='train', crops_dir=get_crop_faces_data_path()):
        super().__init__()
        self.data = data
        self.mode = mode
        self.crops_dir = crops_dir

        self.lookup_table = self._generate_loopup_table(self.mode)

        self.data_len = len(self.data)

        print(f'in init len = {self.data_len}')

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        video_filepath = self.data[index]
        video_filename = os.path.basename(video_filepath)
        video_id = os.path.splitext(video_filename)[0]
        crops_id_path = os.path.join(self.crops_dir, video_id)
        frames = glob(crops_id_path + '/*_0.png')
        frames = sorted(frames, key=alpha_sort_keys)
        item = (video_id, frames, self.lookup_table[video_filename])

        return item

    def _generate_loopup_table(self, mode):
        if mode == 'train':
            labels_csv = get_train_labels_csv_filepath()
        elif mode == 'valid':
            labels_csv = get_valid_labels_csv_filepath()
        elif mode == 'test':
            labels_csv = get_test_labels_csv_filepath()
        else:
            raise Exception("Invalid mode in DFDCDataLoader")
        df = pd.read_csv(labels_csv, squeeze=True, index_col=0)
        df_dict = df.to_dict()
        return df_dict
