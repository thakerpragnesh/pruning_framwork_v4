#!/usr/bin/env python
# coding: utf-8
"""
Iterative structured channel (filter) pruning using a K-means redundancy
criterion: in each conv layer, output channels are clustered by similarity
and, within each cluster, every channel except the largest-L1-norm one is
considered redundant and pruned (see
facilitate_pruning.compute_kmeans_similarity_score_channel). The distance
metric used for clustering is configurable via config.ini's
[Pruning] distance_metric (manhattan/euclidean/cosine); manhattan was the
best-performing metric in evaluation. Each round prunes the *current*
(already pruned + fine-tuned) model, then fine-tunes the resulting smaller
network, so pruning decisions in round N+1 are based on the weights
actually trained in round N.
"""

# In[1]: Import all the required modules
import configparser
import functools
import os
from datetime import date

import torch

import facilitate_pruning as fp
import load_dataset as dl
import load_model as lm
import pruning_driver as pd

# In[2]: Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

dir_home_path = config.get('Dataset', 'dir_home_path')
selected_dataset_dir = config.get('Dataset', 'selected_dataset_dir')
train_folder = config.get('Dataset', 'train_folder')
test_folder = config.get('Dataset', 'test_folder')

load_model_flag = config.getboolean('Model', 'loadModel')
is_transfer_learning = config.getboolean('Model', 'is_transfer_learning')
program_name = config.get('Model', 'program_name')
selected_model = config.get('Model', 'selectedModel')
num_classes = config.getint('Model', 'num_classes')

max_pruning_ratio = config.getfloat('Pruning', 'max_pruning_ratio')
prune_epochs = config.getint('Pruning', 'prune_epochs')
fine_tune_epochs = config.getint('Pruning', 'fine_tune_epochs')
batch_size = config.getint('Pruning', 'batch_size')
image_size = config.getint('Pruning', 'image_size')
max_lr = config.getfloat('Pruning', 'max_lr')
weight_decay = config.getfloat('Pruning', 'weight_decay')
l1_lambda = config.getfloat('Pruning', 'l1_lambda')
grad_clip = config.getfloat('Pruning', 'grad_clip')
distance_metric = config.get('Pruning', 'distance_metric', fallback='manhattan')

# In[3]: Paths
dataset_dir = f"{dir_home_path}Dataset/"
dir_specific_path = f"{program_name}/{selected_dataset_dir}"
model_dir = f"{dir_home_path}Model/{dir_specific_path}"
log_dir = f"{dir_home_path}Logs/{dir_specific_path}"
load_path = f"{model_dir}/{selected_model}"

logResultFile = f'{log_dir}/result.log'
outFile = f'{log_dir}/lastResult.log'
outLogFile = f'{log_dir}/outLogFile.log'


def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir(f'{model_dir}/')
ensure_dir(f'{log_dir}/')

today = date.today()
d1 = today.strftime("%d-%m")


def log(message):
    with open(outLogFile, 'a') as f:
        f.write(message)


log(f"\n\n..........................OutLog For the {d1}......................\n\n")

# In[4]: Cuda device
device1 = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
opt_func = torch.optim.Adam

# In[5]: Data loaders
dl.set_image_size(image_size)
dl.set_batch_size(batch_size)
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir,
                             selected_dataset_arg=selected_dataset_dir,
                             train_arg=train_folder, test_arg=test_folder)

# In[6]: Load model. Uses the batch_norm variant so the source model's
# `.features` layer layout matches the batch_norm=True models rebuilt after
# every pruning round (lm.create_vgg_from_feature_list(..., batch_norm=True))
# -- otherwise their layer indices don't correspond and deep_model_copy_channelwise
# copies into the wrong layers entirely.
if load_model_flag:
    current_model = torch.load(load_path, map_location=device1)
else:
    current_model = lm.load_model(model_name='vgg16bn', number_of_class=num_classes,
                                  pretrainval=is_transfer_learning,
                                  freeze_feature_arg=False, device_l=device1)

# In[7]: Bind the configured distance metric so the function matches the
# score_fn(weight, prune_amount=k) contract every other criterion uses.
kmeans_score_fn = functools.partial(
    fp.compute_kmeans_similarity_score_channel, distance=distance_metric)


# In[8]: Bundle everything prune_one_round needs for every round of this run.
round_kwargs = dict(
    max_pruning_ratio=max_pruning_ratio, fine_tune_epochs=fine_tune_epochs,
    max_lr=max_lr, weight_decay=weight_decay, l1_lambda=l1_lambda, grad_clip=grad_clip,
    opt_func=opt_func, dataloaders=dataLoaders, train_dir=dl.train_directory,
    test_dir=dl.test_directory, device=device1, model_name='vgg16bn',
    log_fn=log, log_file=logResultFile, out_file=outFile, model_dir=model_dir,
    save_prefix='vgg16_IntelIc_Prune',
)


if __name__ == '__main__':
    current_model = pd.iterative_channel_pruning(
        current_model, prune_epochs, kmeans_score_fn, **round_kwargs)
