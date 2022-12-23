# NV project 

Implementation of HiFiGAN for neural vocoder HW.

## Installation guide

```shell
pip install -r ./requirements.txt
python3 load_data.py 
```

## Training

To reproduce training run:

```shell
python3 train.py -c hw_nv/configs/train_run_1.json
```

## Testing

Download model from https://drive.google.com/file/d/1v0OHE4I1kJ6SwcEoky_r0FPwDXa9SX2Q/view?usp=sharing to default_test_model/checkpoint.pth 

To test model (will generate all the needed audio samples for calculating the MOS, if you want to use different sentences just replace wavs in folder test_audio/ and give new wavs same names (or rename their names in test.py)):
```shell
python3 test.py -c hw_nv/configs/train_run_1.json -r default_test_model/checkpoint.pth
```

Generated wavs will be in folder ./results/


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

