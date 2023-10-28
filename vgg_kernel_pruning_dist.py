#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:49:43 2022
@author: pragnesh
"""

# !/usr/bin/env python
# coding: utf-8
# In[1]: import different libraries
# import numpy as np                          # Basic Array and Numeric Operation
# import tarfile                              # use to extract dataset from zip files
# import sys
# import zipfile                              # to extract zip file
# import torchvision                          # Provides facilities to access image dataset

import torch  # Provides basic tensor operation and nn operation
import load_dataset as dl  # create data loader for selected dataset
import load_model as lm  # facilitate loading and manipulating models
import train_model as tm  # Facilitate training of the model
import initialize_pruning as ip  # Initialize and provide basic parameter require for pruning
import facilitate_pruning as fp  # Compute Pruning Value and many things
import torch.nn.utils.prune as prune
import os  # use to access the files
from datetime import date


# In[2] Initialize all the string variable:String parameter for dataset
dataset_dir = '/home/pragnesh/project/Dataset/'; selected_dataset_dir = 'IntelIC'
train_folder = 'train'; test_folder = 'test'

# String Parameter for Model
loadModel = False; is_transfer_learning = False
program_name = 'vgg_net_kernel_pruning_3Aug'; model_dir = '/home/pragnesh/project/Model/';
selectedModel = 'vgg16_IntelIc_Prune'
load_path = f'{model_dir}{program_name}/{selected_dataset_dir}/{selectedModel}'

# String parameter to Log Output
logDir = '/home/pragnesh/project/Logs/'
folder_loc = f'{logDir}{program_name}/{selected_dataset_dir}/'
logResultFile = f'{folder_loc}/result.log'; outFile = f'{folder_loc}/lastResult.log';
outLogFile = f'{folder_loc}/outLogFile.log'

# Check for Cuda Enabled Device
if torch.cuda.is_available():
    device1 = torch.device('cuda')
else:
    device1 = torch.device('cpu')


# In[3]: Ensure require directory for output is present.
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Ensure program name subdirectory is present inside model dir. ~/model_dir/program name/
ensure_dir(f'{model_dir}{program_name}/')
# Ensure selected dataset subdirectory is present inside program name dir ~/model_dir/program_name/selected_dataset
ensure_dir(f'{model_dir}{program_name}/{selected_dataset_dir}/')
# Ensure program name subdirectory is present inside Log Dir ~/logDir/program_name/
ensure_dir(f'{logDir}{program_name}')
# Ensure selected_dataset subdirectory is present inside program_name ~/logDir/program_name/selected_dataset
ensure_dir(f'{logDir}{program_name}/{selected_dataset_dir}/')

# In[4]: Set the data loader to load the data for selected dataset

# set the image properties
dl.set_image_size(224)
dl.set_batch_size = 16
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir,
                             selected_dataset_arg=selected_dataset_dir,
                             train_arg=train_folder,
                             test_arg=test_folder)

# In[5]:load the saved model if we have any and if we don't load a standard model

if loadModel:  # Load the saved trained model
    new_model = torch.load(load_path, map_location=torch.device(device1))
else:  # Load the standard model from library
    # if we don't have any saved trained model download pretrained model for transfer learning
    new_model = lm.load_model(model_name='vgg16', number_of_class=6, pretrainval=is_transfer_learning,
                              freeze_feature_arg=False, device_l=device1)

# In[6]: Change the print statement to write and store the intermediate result in selected file


today = date.today()
d1 = today.strftime("%d-%m")
print(f"\n...........OutLog For the {d1}................")
with open(outLogFile, 'a') as f:
    f.write(f"\n\n..........................OutLog For the {d1}......................\n\n")
f.close()

# In[7]: Initialize all the list and parameter
block_list = []  # ip.getBlockList('vgg16')
feature_list = []
conv_layer_index = []
module = []
prune_count = []
new_list = []
layer_number = 0
st = 0
en = 0
candidate_conv_layer = []


def initialize_lists_for_pruning():
    global block_list, feature_list, conv_layer_index, module, prune_count, new_list, \
        layer_number, st, en, candidate_conv_layer

    with open(outLogFile, "a") as output_file:
        block_list = ip.create_block_list(new_model)  # ip.getBlockList('vgg16')
        feature_list = ip.create_feature_list(new_model)
        conv_layer_index = ip.find_conv_index(new_model)
        module = ip.make_list_conv_param(new_model)
        prune_count = ip.get_prune_count(module=module, blocks=block_list, max_pr=.1)
        new_list = []
        layer_number = 0
        st = 0
        en = 0
        candidate_conv_layer = []

        output_file.write(f"\nBlock List   = {block_list}"
                          f"\nFeature List = {feature_list}"
                          f"\nConv Index   = {conv_layer_index}"
                          f"\nPrune Count  = {prune_count}"
                          f"\nStart Index  = {st}"
                          f"\nEnd Index    = {en}"
                          f"\nInitial Layer Number = {layer_number}"
                          f"\nEmpty candidate layer list = {candidate_conv_layer}"
                          )
        output_file.close()


# ........................... Kernel Pruning .........................................
def compute_conv_layer_saliency_kernel_pruning(module_candidate_convolution, block_list_l, block_id, k=1):
    return module_candidate_convolution + block_list_l + block_id + k


# In[8]: Computer candidate convolution layer  Block-wise
def compute_conv_layer_dist_kernel_pruning(module_candidate_convolution, block_list_l, block_id):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nExecuting Compute Candidate Convolution Layer")
    out_file.close()
    global layer_number
    candidate_convolution_layer = []
    end_index = 0
    for bl in range(len(block_list_l)):
        start_index = end_index
        end_index = end_index + block_list_l[bl]
        if bl != block_id:
            continue
        with open(outLogFile, "a") as out_file:
            out_file.write(f'\nblock ={bl} blockSize={block_list_l[bl]}, start={start_index}, End={end_index}')
        out_file.close()

        # newList = []
        # candidList = []
        for lno in range(start_index, end_index):
            # layer_number =st+i
            with open(outLogFile, 'a') as out_file:
                out_file.write(f"\nlno in compute candidate {lno}")
            out_file.close()
            candidate_convolution_layer.append(fp.compute_distance_score_kernel(
                                                    module_candidate_convolution[lno]._parameters['weight'],
                                                    n=1,
                                                    dim_to_keep=[0, 1],
                                                    prune_amount=prune_count[lno]))
        break
    return candidate_convolution_layer


# In[9]: Extract k_kernel elements form candidate conv layer and store them in the new_list
def compute_new_list(candidate_convolution_layer, k_kernel):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nExecuting Compute New List")
    out_file.close()
    new_list_l = []
    for i in range(len(candidate_convolution_layer)):  # Layer number
        in_channel_list = []
        for j in range(len(candidate_convolution_layer[i])):  # Input channel
            tuple_list = []
            for k in range(k_kernel):  # extract k kernel working on each input channel
                tuple_list.append(candidate_convolution_layer[i][j][k])
            in_channel_list.append(tuple_list)
        new_list_l.append(in_channel_list)
    return new_list_l


# In[10]:Define Custom Pruning
class KernelPruningSimilarities(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        with open(outLogFile, "a") as log_file:
            log_file.write("\n Executing Compute Mask")
        log_file.close()
        mask = default_mask.clone()
        # mask.view(-1)[::2] = 0
        size = t.shape
        print(f"\n{size}")
        with open(outLogFile, "a") as log_file:
            log_file.write(f'\nLayer Number:{layer_number} \nstart={st} \nlength of new list={len(new_list)}')
        log_file.close()
        for k1 in range(len(new_list)):
            for k2 in range(len(new_list[layer_number - st][k1])):
                i = new_list[layer_number - st][k1][k2][1]
                j = new_list[layer_number - st][k1][k2][0]
                if k1 == j:
                    print(":", end='')
                # print(f"i= {i} , j= {j}")

                mask[i][j] = 0
        return mask


def kernel_unstructured_similarities(kernel_module, name):
    KernelPruningSimilarities.apply(kernel_module, name)
    return kernel_module


class ChannelPruningSaliency(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):

        return


# In[11]: Update the feature list to create new temporary model work as compress prune model
def update_feature_list(feature_list_l, prune_count_update, start=0, end=len(prune_count)):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nupdate the feature list")
    out_file.close()
    j = 0
    i = start
    while j < end:
        if feature_list_l[i] == 'M':
            i += 1
            continue
        else:
            feature_list_l[i] = feature_list_l[i] - prune_count_update[j]
            j += 1
            i += 1
    return feature_list_l


# In[12]: copy the non-zero elements form pruned model in the compress temp model
def deep_copy(destination_model, source_model):
    with open(outLogFile, "a") as out_file:
        out_file.write("\n.............Deep Copy Started.........")
    out_file.close()
    print("deepcopy starter")
    for i in range(len(source_model.features)):
        # print(".",end="")
        if str(source_model.features[i]).find('Conv') != -1:
            print("\n   value :", i, "conv value", str(source_model.features[i]).find('Conv'))
            size_org = source_model.features[i]._parameters['weight'].shape
            size_new = destination_model.features[i]._parameters['weight'].shape
            for fin_org in range(size_org[1]):
                j = 0
                fin_new = fin_org
                for fout in range(size_org[0]):
                    if torch.norm(source_model.features[i]._parameters['weight'][fout][fin_org]) != 0:
                        fin_new += 1
                        if j >= size_new[0] or fin_new >= size_new[1]:
                            break
                        t = source_model.features[i]._parameters['weight'][fout][fin_org]
                        destination_model.features[i]._parameters['weight'][j][fin_new] = t
                        j = j + 1


# In[13]: Set the location of the log files to record the results
with open(logResultFile, 'a') as f:
    f.write(f'log for the execution on {d1}')
f.close()

# In[14]: Perform iterative pruning


opt_func = torch.optim.Adam


def iterative_kernel_pruning_dist_block_wise(new_model_arg, prune_module, block_list_l, prune_epochs):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nPruning Process Start")
    out_file.close()
    # pc = [1, 3, 9, 26, 51]
    global new_list
    for e in range(prune_epochs):
        # 1.  Initialization: blockList,featureList,conv_idx,prune_count,module
        # layerIndex=0
        start = 0
        end = len(block_list_l)

        for blkId in range(start, end):
            # 2 Compute the distance between kernel for candidate convolution layer
            new_list = compute_conv_layer_dist_kernel_pruning(module_candidate_convolution=prune_module,
                                                              block_list_l=block_list_l,
                                                              block_id=blkId)

            # 5 perform Custom pruning where we mask the prune weight
            for j in range(block_list_l[blkId]):
                if blkId < 2:
                    layer_number_to_prune = (blkId * 2) + j
                else:  # blkId >= 2:
                    layer_number_to_prune = 4 + (blkId - 2) * 3 + j
                kernel_unstructured_similarities(kernel_module=prune_module[layer_number_to_prune], name='weight')
            new_list = None

        # 6.  Commit Pruning
        with open(outLogFile, 'a') as out_file:
            out_file.write("\ncommit the pruning")
        out_file.close()
        for i in range(len(prune_module)):
            prune.remove(module=prune_module[i], name='weight')

        # 7.  Update feature list
        global feature_list
        feature_list = update_feature_list(feature_list, prune_count, start=0, end=len(prune_count))

        # 8.  Create new temp model with updated feature list
        temp_model = lm.create_vgg_from_feature_list(vgg_feature_list=feature_list,
                                                     batch_norm=True)
        temp_model.to(device1)

        # 9.  Perform deep copy
        lm.freeze(temp_model, 'vgg16')
        deep_copy(temp_model, new_model_arg)
        lm.unfreeze(temp_model)

        # 10.  Train pruned model
        with open(outLogFile, 'a') as out_file:
            out_file.write('\n ...Deep Copy Completed...')
            out_file.write('\n Fine tuning started....')
        out_file.close()

        tm.fit_one_cycle(  # set locations of the dataset, train and test data
            dataloaders=dataLoaders, train_dir=dl.train_directory, test_dir=dl.test_directory,
            # Select a variant of VGGNet
            model_name='vgg16', model=temp_model, device_l=device1,
            # Set all the Hyper-Parameter for training
            epochs=8, max_lr=0.001, weight_decay=0.01, L1=0.01, grad_clip=0.1,
            opt_func=opt_func,
            log_file=logResultFile)

        with open(outLogFile, 'a') as out_file:
            out_file.write('....Fine tuning completed\n')
        out_file.close()

        save_path = f'{model_dir}{program_name}/{selected_dataset_dir}/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)

        # # # 10. Evaluate the pruned model
        train_accuracy = 0.0
        test_accuracy = 0.0

        # train_accuracy = tm.evaluate(newModel, dataLoaders[dl.train_directory], device=device1)
        # _  = tm.evaluate(newModel, dataLoaders[dl.test_directory], device=device1)

        with open(outFile, 'a') as out_file:
            out_file.write(f'\n output of the {e}th iteration is written below\n ')
            out_file.write(f'\n Train Accuracy :  {train_accuracy}\n Test Accuracy  :  {test_accuracy} \n')
        out_file.close()

        save_path = f'{model_dir}{program_name}/selected/dataset_dir/vgg16_IntelIc_Prune_{e}_b_train'
        # save_path = f'/home3/pragnesh/Model/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)


''' Initialize pruning will initialize all the data structure required '''
initialize_lists_for_pruning()
iterative_kernel_pruning_dist_block_wise(new_model_arg=new_model, prune_module=module, block_list_l=block_list, prune_epochs=6)
