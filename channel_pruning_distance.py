#!/usr/bin/env python
# coding: utf-8
"""
Iterative structured channel (filter) pruning using a redundancy /
similarity criterion: in each conv layer, the output channels that are
closest (by normalized distance) to some other channel are considered
redundant and are pruned first. Each round prunes the *current* (already
pruned + fine-tuned) model, then fine-tunes the resulting smaller network,
so pruning decisions in round N+1 are based on the weights actually trained
in round N.
"""

# In[1]: Import all the required modules
import configparser
import os
from datetime import date

import torch
import torch.nn.utils.prune as prune

import facilitate_pruning as fp
import initialize_pruning as ip
import load_dataset as dl
import load_model as lm
import train_model as tm

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


# In[7]: Custom pruning method - zeroes whole output channels (structured
# channel pruning), driven by an externally-supplied list of channel indices.
_channels_to_prune = []


class ChannelPruningMask(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        for ch in _channels_to_prune:
            mask[ch] = 0
        return mask


def prune_channels(conv_module, name, channel_indices):
    global _channels_to_prune
    _channels_to_prune = channel_indices
    ChannelPruningMask.apply(conv_module, name)
    return conv_module


# In[8]: Compute, for every conv layer inside a given block, the list of
# output-channel indices that should be pruned.
def compute_candidate_channels_for_block(score_fn, conv_modules, block_list_l, block_id, prune_count_l):
    candidate_layers = []
    end_index = 0
    for bl, block_size in enumerate(block_list_l):
        start_index = end_index
        end_index = end_index + block_size
        if bl != block_id:
            continue
        for lno in range(start_index, end_index):
            weight = conv_modules[lno]._parameters['weight']
            candidate_layers.append(score_fn(weight, prune_amount=prune_count_l[lno]))
        break
    return candidate_layers


def layer_number_to_prune(block_id, j):
    if block_id < 2:
        return (block_id * 2) + j
    return 4 + (block_id - 2) * 3 + j


# In[9]: Update the feature list after a pruning round
def update_feature_list(feature_list_l, prune_count_update, start=0, end=None):
    if end is None:
        end = len(prune_count_update)
    j = 0
    i = start
    while j < end:
        if feature_list_l[i] == 'M':
            i += 1
            continue
        feature_list_l[i] = feature_list_l[i] - prune_count_update[j]
        j += 1
        i += 1
    return feature_list_l


# In[10]: Main iterative pruning loop
def iterative_channel_pruning_distance_block_wise(model, prune_rounds):
    log("\nPruning Process Start")
    global current_model

    for e in range(prune_rounds):
        block_list = ip.create_block_list(model)
        conv_modules = ip.make_list_conv_param(model)
        prune_count = ip.get_prune_count(module=conv_modules, blocks=block_list, max_pr=max_pruning_ratio)
        feature_list = ip.create_feature_list(model)

        for blk_id in range(len(block_list)):
            candidates = compute_candidate_channels_for_block(
                fp.compute_distance_score_channel, conv_modules, block_list, blk_id, prune_count)
            for j in range(block_list[blk_id]):
                lno = layer_number_to_prune(blk_id, j)
                prune_channels(conv_modules[lno], 'weight', candidates[j])

        for conv_module in conv_modules:
            prune.remove(module=conv_module, name='weight')

        feature_list = update_feature_list(feature_list, prune_count)

        temp_model = lm.create_vgg_from_feature_list(vgg_feature_list=feature_list, batch_norm=True)
        temp_model.to(device1)

        lm.freeze(temp_model)
        fp.deep_model_copy_channelwise(model, temp_model, feature_list)
        lm.unfreeze(temp_model)

        log('\n ...Deep Copy Completed...')
        log('\n Fine tuning started....')

        tm.fit_one_cycle(dataloaders=dataLoaders,
                         train_dir=dl.train_directory, test_dir=dl.test_directory,
                         model_name='vgg16bn', model=temp_model, device_l=device1,
                         epochs=fine_tune_epochs, max_lr=max_lr, weight_decay=weight_decay,
                         L1=l1_lambda, grad_clip=grad_clip,
                         opt_func=opt_func, log_file=logResultFile)

        train_result = tm.evaluate(temp_model, dataLoaders[dl.train_directory])
        test_result = tm.evaluate(temp_model, dataLoaders[dl.test_directory])

        with open(outFile, 'a') as out_file:
            out_file.write(f'\n output of the {e}th iteration is written below\n')
            out_file.write(f'\n Train Accuracy: {train_result["val_acc"]}'
                           f'\n Test Accuracy  : {test_result["val_acc"]} \n')

        save_path = f'{model_dir}/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)

        # Next round prunes and fine-tunes the model this round just produced.
        model = temp_model
        current_model = temp_model


if __name__ == '__main__':
    iterative_channel_pruning_distance_block_wise(current_model, prune_epochs)
