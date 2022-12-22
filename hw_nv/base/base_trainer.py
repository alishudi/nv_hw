from abc import abstractmethod

import torch
from numpy import inf

from hw_nv.base import BaseModel
from hw_nv.logger import get_visualizer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model_gen: BaseModel, model_disc: BaseModel, criterion_gen, criterion_disc,
                optimizer_gen, optimizer_disc, config, device):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model_gen = model_gen
        self.model_disc = model_disc
        self.criterion_gen = criterion_gen
        self.criterion_disc = criterion_disc
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        if config.load is not None:
            self._load_checkpoint(config.load)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving models on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate models performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch_gen = type(self.model_gen).__name__
        arch_disc = type(self.model_disc).__name__
        state = {
            "arch_gen": arch_gen,
            "arch_disc": arch_disc,
            "epoch": epoch,
            "state_dict_gen": self.model_gen.state_dict(),
            "state_dict_disc": self.model_disc.state_dict(),
            "optimizer_gen": self.optimizer_gen.state_dict(),
            "optimizer_disc": self.optimizer_disc.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config.config,
        }
        #hard to define best for GANs, so i just save last (dont want to save all, cause they weight a lot)
        filename = str(self.checkpoint_dir / "checkpoint-last.pth")
        config_path = str (self.checkpoint_dir / "config.json")
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        self.writer.add_checkpoint(filename, config_path)


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model_gen.load_state_dict(checkpoint["state_dict_gen"])
        self.model_disc.load_state_dict(checkpoint["state_dict_disc"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer_gen"] != self.config["optimizer_gen"] or
                checkpoint["config"]["optimizer_disc"] != self.config["optimizer_disc"] or
                checkpoint["config"]["lr_scheduler_gen"] != self.config["lr_scheduler_gen"] or
                checkpoint["config"]["lr_scheduler_disc"] != self.config["lr_scheduler_disc"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer_gen.load_state_dict(checkpoint["optimizer_gen"])
            self.optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    def _load_checkpoint(self, load_path):
        """
        Load model from saved checkpoints

        :param load_path: Checkpoint path to be loaded
        """
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path, self.device)
        # self.mnt_best = checkpoint["monitor_best"] 

        # load architecture params from checkpoint.
        if (
            checkpoint["config"]["arch"] != self.config["arch"]
        ):
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model_gen.load_state_dict(checkpoint["state_dict_gen"])
        self.model_disc.load_state_dict(checkpoint["state_dict_disc"])

        self.logger.info(
            "Checkpoint loaded."
        )
