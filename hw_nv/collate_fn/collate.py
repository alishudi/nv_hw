import logging
from typing import List
import torch
import numpy as np
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def pad_1D_tensor(inputs):

    def pad_data(x, length):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len) for x in inputs])

    return padded


def pad_2D_tensor(inputs):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    max_len = max(x.size(0) for x in inputs)
    output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["src_seq"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration_predictor_target"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    energies = pad_1D_tensor(energies)
    pitches = pad_1D_tensor(pitches)

    out = {"src_seq": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
           "energy": energies,
           "pitch": pitches
           }

    return out



def collate_fn(dataset_items: List[dict], batch_expand_size=1):
    """
    Collate and pad fields in dataset items
    """
    len_arr = np.array([d["src_seq"].size(0) for d in dataset_items])
    index_arr = np.argsort(-len_arr)
    batchsize = len(dataset_items)
    real_batchsize = batchsize // batch_expand_size

    cut_list = list()
    for i in range(batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = {
        "src_seq": [],
        "mel_target": [],
        "duration": [],
        "mel_pos": [],
        "src_pos": [],
        "mel_max_len": [],
        "energy": [],
        "pitch": []
        }
    for i in range(batch_expand_size):
        out = (reprocess_tensor(dataset_items, cut_list[i]))
        for name in ["src_seq", "mel_target", "duration", "mel_pos", "src_pos", "mel_max_len", "energy", "pitch"]:
            output[name].append(out[name])

    return output