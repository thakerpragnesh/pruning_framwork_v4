#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import load_model as lm
# import torch

# In[2]:
vgg11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
vgg13 = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
vgg16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
vgg19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],


# In[3]:
def get_block_list(model_name):
    blocks = []
    if model_name == 'vgg11':
        blocks = [1, 1, 2, 2, 2]
    if model_name == 'vgg11bn':
        blocks = [1, 1, 2, 2, 2]  # prune_list = [0, 3, 6,8, 11,13, 16,18]
    if model_name == 'vgg13':
        blocks = [2, 2, 2, 2, 2]
    if model_name == 'vgg13bn':
        blocks = [2, 2, 2, 2, 2]
    if model_name == 'vgg16':
        blocks = [2, 2, 3, 3, 3]
    if model_name == 'vgg16bn':
        blocks = [2, 2, 3, 3, 3]
    return blocks


# In[4]:
def create_block_list(new_model):
    block_list = []
    count = 0
    for i in range(len(new_model.features)):
        if str(new_model.features[i]).find('Conv') != -1:
            count += 1
        elif str(new_model.features[i]).find('Pool') != -1:
            block_list.append(count)
            count = 0
    return block_list


# In[5]: Indices of conv layer in vgg11/13/16 are store in this list
def get_conv_index(model_name):
    feature_list = []
    if model_name == 'vgg11':
        feature_list = [0, 3, 6, 8, 11, 13, 16, 18]
    if model_name == 'vgg11bn':
        feature_list = [0, 4, 8, 11, 15, 18, 22, 25]
    if model_name == 'vgg13':
        feature_list = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22]
    if model_name == 'vgg13bn':
        feature_list = [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]
    if model_name == 'vgg16':
        feature_list = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    if model_name == 'vgg16bn':
        feature_list = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    return feature_list


# In[6]:
def find_conv_index(new_model):
    conv_list_idx = []
    for i in range(len(new_model.features)):
        if str(new_model.features[i]).find('Conv') != -1:
            conv_list_idx.append(i)
    return conv_list_idx


# In[7]:
def get_feature_list(model_name):
    if model_name == 'vgg11':
        return vgg11

    if model_name == 'vgg13':
        return vgg13

    if model_name == 'vgg16':
        return vgg16


# In[8]:
def create_feature_list(new_model):
    feature_list = []
    for i in range(len(new_model.features)):
        if str(new_model.features[i]).find('Conv') != -1:
            size = new_model.features[i]._parameters['weight'].shape
            n = size[0]
            feature_list.append(n)
        if str(new_model.features[i]).find('Pool') != -1:
            feature_list.append('M')
    # feature_list.pop(-1)
    return feature_list


# In[9]: Create a list that contain all the conv layer
def make_list_conv_param(new_model):
    conv_list = []
    for i in range(len(new_model.features)):
        if str(new_model.features[i]).find('Conv') != -1:
            conv_list.append(new_model.features[i])
    return conv_list


# In[10]: create a prune count list which prepare a list of number of channel to prune
# from each list from max_pruning_ratio
def get_prune_count(module, blocks, max_pr):
    j = 0
    count = 0
    prune_prob = []
    prune_count = []
    step = max_pr / len(blocks)
    p = step
    for i in range(len(module)):
        if count >= blocks[j]:
            p = p + step
            count = 0
            j = j + 1
        prune_prob.append(p)
        count += 1

    for i in range(len(module)):
        size = module[i]._parameters['weight'].shape
        c = int(round(size[0] * prune_prob[i]))
        prune_count.append(c)
    return prune_count
