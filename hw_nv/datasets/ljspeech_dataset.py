import logging
import os

from hw_nv.utils import ROOT_PATH

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import time
from torch.utils.data import Dataset
from hw_nv.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class LJspeechDataset(Dataset):
    def __init__(self, config_parser: ConfigParser, segment_size):
        self.data_dir = ROOT_PATH / "data" / "wavs"
        self.buffer = self.get_data_to_buffer()
        self.length_dataset = len(self.buffer)
        self.segment_size = segment_size

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        wav = self.buffer[idx]['true_wavs']
        if wav.shape[1] > self.segment_size:
            l = np.random.randint(0, wav.shape[1] - self.segment_size)
            r = l + self.segment_size
            wav = wav[:,l:r]
        elif wav.shape[1] < self.segment_size:
            wav = F.pad(wav.squeeze(dim=0), (self.segment_size - wav.shape[1]))
        return {'true_wavs': wav}

    def get_data_to_buffer(self):
        buffer = list()
        start = time.perf_counter()
        print("Loading data to the buffer")

        for filename in os.listdir(str(self.data_dir)):
            waveform, sr = torchaudio.load(str(self.data_dir / filename))
            buffer.append({"true_wavs": waveform})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))

        return buffer