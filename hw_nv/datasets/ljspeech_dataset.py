import logging
import os

from hw_nv.utils import ROOT_PATH

import torch
import time
from torch.utils.data import Dataset
from hw_nv.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class LJspeechDataset(Dataset):
    def __init__(self, config_parser: ConfigParser):
        self.data_dir = ROOT_PATH / "data" / "mels"
        self.buffer = self.get_data_to_buffer()
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        #returning random segment of melspec so batch can fit in GPU
        tmp = self.buffer[idx]
        l = torch.randint(0, tmp['mel'].shape[-1] - SEGMENT_SIZE)
        r = l + SEGMENT_SIZE
        tmp['mel'] = tmp['mel'][:,:,l:r]
        return tmp

    def get_data_to_buffer(self):
        buffer = list()
        start = time.perf_counter()
        print("Loading data to the buffer")

        for filename in os.listdir(str(self.data_dir)):
            mel = torch.load(filename)
            buffer.append({"mel": mel})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))

        return buffer