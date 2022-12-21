import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_nv.model as module_model
from hw_nv.trainer import Trainer
from hw_nv.utils import ROOT_PATH
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser
from hw_nv.utils import get_WaveGlow
from hw_nv.synthesis import synthesis
from hw_nv.logger import get_visualizer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    # logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    #loading pretrained WaveGlow model
    waveglow = get_WaveGlow()
    waveglow = waveglow.cuda()

    #TODO add custom test sample support
    test_samples = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education", 
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
            ]
    encoded_test = list(text_to_sequence(test, ['english_cleaners']) for test in test_samples)

    logger = config.get_logger("trainer")
    cfg_trainer = config["trainer"]
    writer = get_visualizer(config, logger, cfg_trainer["visualize"])

    with torch.no_grad():
        for i, phn in tqdm(enumerate(encoded_test)):
                mel, path = synthesis(model, phn, device, waveglow, i)
                name = f'track={i} speed={1} energy={1} pitch={1}'
                writer.add_audio('audio ' + name, path, sample_rate=22050)
        for energy in [0.8, 1.2]:
            for i, phn in tqdm(enumerate(encoded_test)):
                mel, path = synthesis(model, phn, device, waveglow, i, alpha_e=energy)
                name = f'track={i} speed={1} energy={energy} pitch={1}'
                writer.add_audio('audio ' + name, path, sample_rate=22050)
        for speed in [0.8, 1.2]:
            for i, phn in tqdm(enumerate(encoded_test)):
                mel, path = synthesis(model, phn, device, waveglow, i, speed=speed)
                name = f'track={i} speed={2-speed} energy={1} pitch={1}'
                writer.add_audio('audio ' + name, path, sample_rate=22050)
        for pitch in [0.8, 1.2]:
            for i, phn in tqdm(enumerate(encoded_test)):
                mel, path = synthesis(model, phn, device, waveglow, i, alpha_p=pitch)
                name = f'track={i} speed={1} energy={1} pitch={pitch}'
                writer.add_audio('audio ' + name, path, sample_rate=22050)
        for three in [0.8, 1.2]:
            for i, phn in tqdm(enumerate(encoded_test)):
                mel, path = synthesis(model, phn, device, waveglow, i, speed=(2-three), alpha_e=three, alpha_p=three)
                name = f'track={i} speed={three} energy={three} pitch={three}'
                writer.add_audio('audio ' + name, path, sample_rate=22050)


    

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))


    # assert config.config.get("data", {}).get("test", None) is not None
    # config["data"]["test"]["batch_size"] = args.batch_size
    # config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
