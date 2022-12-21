from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa

import os
import shutil
from speechbrain.utils.data_utils import download_file
from hw_nv.utils import ROOT_PATH

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}
# SEGMENT_SIZE = 8192 #took from authors impementation

def load_ljspeech(data_dir):
    #loads wavs into "data_dir"/wavs
    arch_path = data_dir / "LJSpeech-1.1.tar.bz2"
    print(f"Loading LJSpeech")
    download_file(URL_LINKS["dataset"], arch_path)
    shutil.unpack_archive(arch_path, data_dir)
    for fpath in (data_dir / "LJSpeech-1.1").iterdir():
        shutil.move(str(fpath), str(data_dir / fpath.name))
    os.remove(str(arch_path))
    shutil.rmtree(str(data_dir / "LJSpeech-1.1"))

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


if __name__ == "__main__":
    #Loading dataset and precalculating melspecs
    #authors dont use precalculation and are taking random segment of waveform before every melspec calculation
    #i will take segments from melspec
    data_dir = ROOT_PATH / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    wavs_dir = data_dir / "wavs"
    mel_dir = data_dir / "mels"
    if not (wavs_dir).exists():
        load_ljspeech(data_dir)

    if not (mel_dir).exists():
        print('precalculating melspecs')
        mel_dir.mkdir(exist_ok=True, parents=True)

        M = MelSpectrogram(MelSpectrogramConfig)
        for filename in os.listdir(str(wavs_dir)):
            waveform, sr = torchaudio.load(str(wavs_dir / filename))
            mel = M(waveform)
            torch.save(mel, str(mel_dir) + f'/{filename[:-4]}.mel') #TODO clear
    print('Done')
            # if waveform.shape[1] > SEGMENT_SIZE:
                # # l = torch.randint(0, waveform.shape[1] - SEGMENT_SIZE)
                # # r = l + SEGMENT_SIZE
                # # waveform = waveform[:,l:r]
                # # torch.save(waveform, str(mel_dir) + f'/{filename[:-4]}.mel')
                # k = waveform.shape[1] // SEGMENT_SIZE
                # for i in range(k):
                #     l = k * SEGMENT_SIZE
                #     r = l + SEGMENT_SIZE
                #     torch.save(waveform[:,l:r], str(mel_dir) + f'/{filename[:-4]}_{k}.mel')

