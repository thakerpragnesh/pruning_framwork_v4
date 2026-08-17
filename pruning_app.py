#!/usr/bin/env python
# coding: utf-8
"""
Shared setup for every channel-pruning driver script (channel_pruning_*.py,
hybrid_pruning.py): config.ini parsing, path/logging setup, CUDA device,
dataloaders, and model loading -- identical across every pruning criterion.

Before this module existed, that setup (~100 lines) was duplicated verbatim
in every driver script, differing only in which score_fn ended up passed to
pruning_driver at the very end. That meant a fix to the shared setup (e.g.
the `distance_metric` config key added for K-means) had to be hand-copied
into every script, and a new criterion meant copy-pasting the whole file.
`PruningEnvironment` owns that setup exactly once; each driver script's only
remaining job is choosing which score_fn(s) to hand to `.run()`/`.run_hybrid()`
-- adding a new criterion is now a ~10-line script, not a ~100-line copy.
"""
import configparser
import os
from datetime import date

import torch

import load_dataset as dl
import load_model as lm
import pruning_driver as pd


class PruningEnvironment:
    def __init__(self, save_prefix, config_path='config.ini'):
        self.save_prefix = save_prefix
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self._read_config()
        self._setup_paths()
        self._setup_logging()
        self._setup_device()
        self._setup_data()
        self._load_model()

    def _read_config(self):
        c = self.config
        self.dir_home_path = c.get('Dataset', 'dir_home_path')
        self.selected_dataset_dir = c.get('Dataset', 'selected_dataset_dir')
        self.train_folder = c.get('Dataset', 'train_folder')
        self.test_folder = c.get('Dataset', 'test_folder')

        self.load_model_flag = c.getboolean('Model', 'loadModel')
        self.is_transfer_learning = c.getboolean('Model', 'is_transfer_learning')
        self.program_name = c.get('Model', 'program_name')
        self.selected_model = c.get('Model', 'selectedModel')
        self.num_classes = c.getint('Model', 'num_classes')

        self.max_pruning_ratio = c.getfloat('Pruning', 'max_pruning_ratio')
        self.prune_epochs = c.getint('Pruning', 'prune_epochs')
        self.fine_tune_epochs = c.getint('Pruning', 'fine_tune_epochs')
        self.batch_size = c.getint('Pruning', 'batch_size')
        self.image_size = c.getint('Pruning', 'image_size')
        self.max_lr = c.getfloat('Pruning', 'max_lr')
        self.weight_decay = c.getfloat('Pruning', 'weight_decay')
        self.l1_lambda = c.getfloat('Pruning', 'l1_lambda')
        self.grad_clip = c.getfloat('Pruning', 'grad_clip')

    def _setup_paths(self):
        self.dataset_dir = f"{self.dir_home_path}Dataset/"
        dir_specific_path = f"{self.program_name}/{self.selected_dataset_dir}"
        self.model_dir = f"{self.dir_home_path}Model/{dir_specific_path}"
        self.log_dir = f"{self.dir_home_path}Logs/{dir_specific_path}"
        self.load_path = f"{self.model_dir}/{self.selected_model}"

        self.log_result_file = f'{self.log_dir}/result.log'
        self.out_file = f'{self.log_dir}/lastResult.log'
        self.out_log_file = f'{self.log_dir}/outLogFile.log'

        self._ensure_dir(f'{self.model_dir}/')
        self._ensure_dir(f'{self.log_dir}/')

    @staticmethod
    def _ensure_dir(dir_path):
        directory = os.path.dirname(dir_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _setup_logging(self):
        today = date.today().strftime("%d-%m")
        self.log(f"\n\n..........................OutLog For the {today}......................\n\n")

    def log(self, message):
        with open(self.out_log_file, 'a') as f:
            f.write(message)

    def _setup_device(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.opt_func = torch.optim.Adam

    def _setup_data(self):
        dl.set_image_size(self.image_size)
        dl.set_batch_size(self.batch_size)
        self.dataloaders = dl.data_loader(
            set_datasets_arg=self.dataset_dir,
            selected_dataset_arg=self.selected_dataset_dir,
            train_arg=self.train_folder, test_arg=self.test_folder)

    def _load_model(self):
        # Uses the batch_norm variant so the source model's `.features` layer
        # layout matches the batch_norm=True models rebuilt after every
        # pruning round (lm.create_vgg_from_feature_list(..., batch_norm=True))
        # -- otherwise layer indices don't correspond and
        # deep_model_copy_channelwise copies into the wrong layers entirely.
        if self.load_model_flag:
            self.model = torch.load(self.load_path, map_location=self.device)
        else:
            self.model = lm.load_model(
                model_name='vgg16bn', number_of_class=self.num_classes,
                pretrainval=self.is_transfer_learning,
                freeze_feature_arg=False, device_l=self.device)

    def _round_kwargs(self):
        return dict(
            max_pruning_ratio=self.max_pruning_ratio, fine_tune_epochs=self.fine_tune_epochs,
            max_lr=self.max_lr, weight_decay=self.weight_decay, l1_lambda=self.l1_lambda,
            grad_clip=self.grad_clip, opt_func=self.opt_func, dataloaders=self.dataloaders,
            train_dir=dl.train_directory, test_dir=dl.test_directory, device=self.device,
            model_name='vgg16bn', log_fn=self.log, log_file=self.log_result_file,
            out_file=self.out_file, model_dir=self.model_dir, save_prefix=self.save_prefix,
        )

    def run(self, score_fn, prune_epochs=None):
        """Run `prune_epochs` rounds (default: config.ini's [Pruning]
        prune_epochs) of a single criterion, e.g. env.run(fp.compute_saliency_score_channel).
        """
        rounds = self.prune_epochs if prune_epochs is None else prune_epochs
        self.model = pd.iterative_channel_pruning(self.model, rounds, score_fn, **self._round_kwargs())
        return self.model

    def run_hybrid(self, stages):
        """Run a hybrid sequence of (score_fn, num_rounds) stages, e.g.
        env.run_hybrid([(fp.compute_max3_saliency_score_channel, 4), ...]).
        """
        self.model = pd.hybrid_channel_pruning(self.model, stages, **self._round_kwargs())
        return self.model
