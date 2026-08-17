#!/usr/bin/env python
# coding: utf-8
"""
Shared driver for iterative structured channel pruning.

Factors out the prune -> shrink -> deep-copy -> fine-tune -> save loop that
used to be duplicated (nearly verbatim) between channel_pruning_distance.py
and channel_pruning_saliency.py, so every channel-pruning criterion
(L1 saliency, L1 distance, Max3 saliency, SVD saliency, K-means similarity,
...) can reuse the same round-by-round mechanics via a swappable `score_fn`,
and so a hybrid strategy can switch `score_fn` mid-run without duplicating
the loop again.

`score_fn` must have the contract used throughout this codebase:
    score_fn(weight_tensor, prune_amount=k) -> list[int]
i.e. given a conv layer's weight tensor, return the indices of the `k`
channels to prune (weakest/most-redundant first), exactly like
facilitate_pruning.compute_saliency_score_channel /
compute_distance_score_channel already do.

This module takes plain arguments/keyword-arguments -- it does not read
config.ini or set up paths itself; each driver script keeps owning its own
config/path wiring.
"""
import torch
import torch.nn.utils.prune as prune

import facilitate_pruning as fp
import initialize_pruning as ip
import load_model as lm
import train_model as tm

# In[1]: Custom pruning method - zeroes whole output channels (structured
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


# In[2]: Compute, for every conv layer inside a given block, the list of
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


# In[3]: Update the feature list after a pruning round
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


# In[4]: Run a single prune -> shrink -> deep-copy -> fine-tune -> save round
# against `model` using `score_fn` as the channel-selection criterion, and
# return the resulting (smaller, fine-tuned) model.
def prune_one_round(model, score_fn, *, max_pruning_ratio, fine_tune_epochs,
                     max_lr, weight_decay, l1_lambda, grad_clip, opt_func,
                     dataloaders, train_dir, test_dir, device, model_name,
                     log_fn, log_file, out_file, model_dir, save_prefix, round_idx):
    log_fn("\nPruning Process Start")

    block_list = ip.create_block_list(model)
    conv_modules = ip.make_list_conv_param(model)
    prune_count = ip.get_prune_count(module=conv_modules, blocks=block_list, max_pr=max_pruning_ratio)
    feature_list = ip.create_feature_list(model)

    for blk_id in range(len(block_list)):
        candidates = compute_candidate_channels_for_block(
            score_fn, conv_modules, block_list, blk_id, prune_count)
        for j in range(block_list[blk_id]):
            lno = layer_number_to_prune(blk_id, j)
            prune_channels(conv_modules[lno], 'weight', candidates[j])

    for conv_module in conv_modules:
        prune.remove(module=conv_module, name='weight')

    feature_list = update_feature_list(feature_list, prune_count)

    temp_model = lm.create_vgg_from_feature_list(vgg_feature_list=feature_list, batch_norm=True)
    temp_model.to(device)

    lm.freeze(temp_model)
    fp.deep_model_copy_channelwise(model, temp_model, feature_list)
    lm.unfreeze(temp_model)

    log_fn('\n ...Deep Copy Completed...')
    log_fn('\n Fine tuning started....')

    tm.fit_one_cycle(dataloaders=dataloaders,
                     train_dir=train_dir, test_dir=test_dir,
                     model_name=model_name, model=temp_model, device_l=device,
                     epochs=fine_tune_epochs, max_lr=max_lr, weight_decay=weight_decay,
                     L1=l1_lambda, grad_clip=grad_clip,
                     opt_func=opt_func, log_file=log_file)

    train_result = tm.evaluate(temp_model, dataloaders[train_dir])
    test_result = tm.evaluate(temp_model, dataloaders[test_dir])

    with open(out_file, 'a') as f:
        f.write(f'\n output of the {round_idx}th iteration is written below\n')
        f.write(f'\n Train Accuracy: {train_result["val_acc"]}'
                f'\n Test Accuracy  : {test_result["val_acc"]} \n')

    save_path = f'{model_dir}/{save_prefix}_{round_idx}_b_train'
    torch.save(temp_model, save_path)

    return temp_model


# In[5]: Run `prune_rounds` rounds using the same criterion throughout.
def iterative_channel_pruning(model, prune_rounds, score_fn, **round_kwargs):
    for round_idx in range(prune_rounds):
        model = prune_one_round(model, score_fn, round_idx=round_idx, **round_kwargs)
    return model


# In[6]: Hybrid pruning - run through a sequence of (score_fn, num_rounds)
# stages, carrying the shrinking model forward across stages. `round_idx`
# runs continuously across stages so checkpoint filenames stay unique.
def hybrid_channel_pruning(model, stages, **round_kwargs):
    round_idx = 0
    for score_fn, num_rounds in stages:
        for _ in range(num_rounds):
            model = prune_one_round(model, score_fn, round_idx=round_idx, **round_kwargs)
            round_idx += 1
    return model
