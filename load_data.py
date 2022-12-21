import os
import shutil
from speechbrain.utils.data_utils import download_file
from hw_nv.utils import ROOT_PATH

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}

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


if __name__ == "__main__":
    #Loading dataset
    data_dir = ROOT_PATH / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    wavs_dir = data_dir / "wavs"
    if not (wavs_dir).exists():
        load_ljspeech(data_dir)

    print('Done')


