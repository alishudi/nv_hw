# NV project 

Implementation of HiFiGAN for neural vocoder HW.

## Installation guide

#todo rewrite

```shell
pip install -r ./requirements.txt

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p pretrained_model/
mkdir -p data/waveglow/
mv waveglow_256channels_ljs_v2.pt pretrained_model/waveglow_256channels.pt
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz

wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
mv pretrained_model data/waveglow/
mv alignments data/
mv mels data/
rm mel.tar.gz
rm alignments.zip
rm LJSpeech-1.1.tar.bz2
```

To precalculate energies and pitches for training:

```shell
python3 calc_energies.py
python3 calc_pitches.py
```
Or you could try to download archives from https://drive.google.com/file/d/1U-bBkdmKvIBjSx0T1O87YwIY-502YitC/view?usp=sharing and https://drive.google.com/file/d/1iaCr54kbZ6RVUBLj-1VffHuSAhIwM5N9/view?usp=sharing and unarchive into the ./data folder

## Testing

Download model from https://drive.google.com/file/d/1eqR-S4Jdzoa812qPylzo8TSaDg1VSTDz/view?usp=sharing to default_test_model/checkpoint.pth 

To test model (will generate all the needed audio samples for calculating the MOS, if you want to use different sentences just replace default sentences in test.py with them):
```shell
python3 test.py -c hw_tts/configs/train_run_1.json -r default_test_model/checkpoint.pth
```


To train the model run:

```shell
python3 train.py -c hw_asr/configs/train_run_1.json
python3 train.py -c hw_asr/configs/train_run_2.json -l <path to the checkpoint-epoch100.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_3.json -l <path to the checkpoint-epoch100.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_4.json -l <path to the checkpoint-epoch100.pth from the last step>
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

