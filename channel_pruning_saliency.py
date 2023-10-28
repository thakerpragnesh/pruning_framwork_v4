#!/usr/bin/env python
# coding: utf-8

# In[1]: Import all the require files
import torch
import load_dataset as dl
import load_model as lm
import train_model as tm
import initialize_pruning as ip
import facilitate_pruning as fp
import torch.nn.utils.prune as prune
import os  # use to access the files
from datetime import date
import configparser

# In[2]: Set date to store information in the logs 
today = date.today()
d1 = today.strftime("%d_%m") #ex "27_11"

# In[3] Set pathe of all the directories
config = configparser.ConfigParser()
# Read the configuration file
config.read('config.ini')

# In[4]: Set parameter related to Dataset
dataset_dir = config.get('Dataset', 'dataset_dir')
selected_dataset_dir = config.get('Dataset', 'selected_dataset_dir')
train_folder = config.get('Dataset', 'train_folder')
test_folder = config.get('Dataset', 'test_folder')

# In[5]: Set parameter related to Model

if config.get('Model','loadModel') == 'True':
    loadModel = True
else:
     loadModel = False
     


if config.get('Model','is_transfer_learning') == 'True':
    is_transfer_learning = True
else:
    is_transfer_learning = False


program_name = config.get('Model','program_name')
#model_dir = config.get('Model','model_dir')
selectedModel = config.get('Model','selectedModel')

dir_home_path = config.get('Dataset','dir_home_path')

# In[6]
dir_specific_path =f'{program_name}/{selected_dataset_dir}'

# Model Paths
model_dir   = f"{dir_home_path}Model/{dir_specific_path}"
loadModel = False
load_path = f'{model_dir}/{selectedModel}'
is_transfer_learning = False

# Dataset Paths
dataset_dir = f"{dir_home_path}Dataset/{selected_dataset_dir}" 
train_folder = 'train'
test_folder = 'test'

# Logs Path
log_dir       = f"{dir_home_path}Logs/{dir_specific_path}" 
logResultFile = f'{log_dir}/result.log'
outFile       = f'{log_dir}/lastResult.log'
outLogFile    = f'{log_dir}/outLogFile.log'


# In[7]: Check Cuda Devices
if torch.cuda.is_available():
    device1 = torch.device('cuda')
else:
    device1 = torch.device('cpu')


# In[8]: Function to create folder if not exist
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir(f'{dir_home_path}Model/{dir_specific_path}/')
ensure_dir(model_dir)
ensure_dir(f'{dir_home_path}Logs/{dir_specific_path}/')
ensure_dir(log_dir)

# In[9]: Set Image Properties
dl.set_image_size(224)
dl.set_batch_size = 16
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir,
                             selected_dataset_arg='',
                             train_arg=train_folder, test_arg=test_folder)

# In[10]: Load appropriate model
if loadModel:  # Load the saved trained model
    new_model = torch.load(load_path, map_location=torch.device(device1))
else:  # Load the standard model from library
    new_model = lm.load_model(model_name='vgg16', number_of_class=6,
                              pretrainval=is_transfer_learning,
                              freeze_feature_arg=False, device_l=device1)

opt_func = torch.optim.Adam
# In[]
print(new_model)


# In[11]: Create require lists for pruning
block_list = []
feature_list = []

prune_count = []
conv_layer_index = []
module = []

new_list = []
candidate_conv_layer = []

layer_number = 0
st = 0
en = 0


def initialize_lists_for_pruning():
    global block_list
    block_list = ip.create_block_list(new_model)  # ip.getBlockList('vgg16')
    
    global feature_list
    feature_list = ip.create_feature_list(new_model)
    
    global module
    module = ip.make_list_conv_param(new_model)
    
    global prune_count
    prune_count = ip.get_prune_count(module=module, blocks=block_list, max_pr=.1)
    
    global conv_layer_index
    conv_layer_index = ip.find_conv_index(new_model)
    
initialize_lists_for_pruning()    


# In[12] Function to update the feature list after pruning
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


# In[]
'''
def compute_saliency_score_channel(tensor_t, n=1, dim_to_keep=[0], prune_amount=1):
    INF = 10000.0
    dim_to_prune = list(range(tensor_t.dim()))
    for i in range(len(dim_to_keep)):
        dim_to_prune.remove(dim_to_keep[i])

    size = tensor_t.shape
    print(size)
    print(dim_to_keep)
    channel_norm = torch.norm(tensor_t, p=1, dim=dim_to_prune)
    channel_norm_temp = torch.norm(tensor_t, p=1, dim=dim_to_prune)
    score_value = []
    for i in range(prune_amount):
        min_idx = 0
        for j in range(size[0]):
            if channel_norm_temp[min_idx] > channel_norm_temp[j]:
                min_idx = j
        score_value.append([min_idx, channel_norm[min_idx]])
        channel_norm_temp[min_idx] = INF

    return score_value
'''
print()

# In[13]:
def compute_conv_layer_saliency_channel_pruning(module_cand_conv, block_list_l, block_id, k=1):
    global layer_number
    candidate_convolution_layer = []
    end_index = 0
    for bl in range(len(block_list_l)):
        start_index = end_index
        end_index = end_index + block_list_l[bl]
        if bl != block_id:
            continue

        for lno in range(start_index, end_index):
            # layer_number =st+i
            list_ele = fp.compute_saliency_score_channel(module_cand_conv[lno]._parameters['weight'],
                                              n=1, 
                                              dim_to_keep=[0], 
                                              prune_amount=prune_count[lno]
                                              )
            
            candidate_convolution_layer.append(list_ele)
                

        break
    return candidate_convolution_layer


# In[14]: Compute mask matrix using saliency score
class ChannelPruningMethodSaliency(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'


    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        global layer_number
        global layer_base
        i=layer_number-layer_base
        for j in range(len(new_list[i])):
            k = new_list[i][j][0]
            print("value of k is:",k)
            #mask[k] =0
            break
            
        return mask
    
def channel_unstructured_saliency(module, name):
    ChannelPruningMethodSaliency.apply(module, name)
    return module


# In[] Deep Copy: Copy nonzero parameter of prune model into new model
'''
def deep_model_copy_channelwise(source_model, destination_model, feature_list):
    for l in range(len(source_model.features)):
        if str(source_model.features[l]).find('Conv') != -1:
            size_org = source_model.features[l]._parameters['weight'].shape
            #size_new = destination_model.features[l]._parameters['weight'].shape
            out_ch_new =0
            for out_ch_old in range(size_org[0]):
                if torch.norm(source_model.features[l]._parameters['weight'][out_ch_old] != 0):
                    t = source_model.features[l]._parameters['weight'][out_ch_old]
                    destination_model.features[l]._parameters['weight'][out_ch_new] = t
                    out_ch_old +=1
                

'''



# In[15]:
layer_base=0
def iterative_channel_pruning_saliency_block_wise(new_model_arg, prune_module, 
                                             block_list_l, prune_epochs):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nPruning Process Start")
    out_file.close()
    # pc = [1, 3, 9, 26, 51]
    
    global new_list
    global layer_base
    
    for e in range(prune_epochs):
        start = 0
        end = len(block_list_l)
        for blkId in range(start, end):
            # 2 Compute distance between kernel for candidate conv layer
            new_list = compute_conv_layer_saliency_channel_pruning(module_cand_conv=prune_module,
                                                                   block_list_l=block_list_l, block_id=blkId)
            # 5 perform Custom pruning where we mask the prune weight
            for j in range(block_list_l[blkId]):
                if blkId < 2:
                    layer_number_to_prune = (blkId * 2) + j
                else:  # blkId >= 2:
                    layer_number_to_prune = 4 + (blkId - 2) * 3 + j
                channel_unstructured_saliency(
                    module=prune_module[layer_number_to_prune], 
                    name='weight')
            new_list = None
        # 6.  Commit Pruning
        for i in range(len(prune_module)):
            prune.remove(module=prune_module[i], name='weight')
        # 7.  Update feature list
        global feature_list
        feature_list = update_feature_list(
            feature_list, prune_count, start=0, end=len(prune_count))
        # 8.  Create new temp model with updated feature list
        temp_model = lm.create_vgg_from_feature_list(
            vgg_feature_list=feature_list, batch_norm=True)
        temp_model.to(device1)
        
        
        # 9.  Perform deep copy
        lm.freeze(temp_model, 'vgg16')
        fp.deep_model_copy_channelwise(new_model, temp_model, feature_list)
        #deep_copy(temp_model, new_model_arg)
        lm.unfreeze(temp_model)
        
        
        # 10.  Train pruned model
        with open(outLogFile, 'a') as out_file:
            out_file.write('\n ...Deep Copy Completed...')
            out_file.write('\n Fine tuning started....')
        out_file.close()

        tm.fit_one_cycle( dataloaders=dataLoaders,
                          train_dir=dl.train_directory, test_dir=dl.test_directory,
                          # Select a variant of VGGNet
                          model_name='vgg16', model=temp_model, device_l=device1,
                          # Set all the Hyper-Parameter for training
                          epochs=8, max_lr=0.001, weight_decay=0.01, L1=0.01, grad_clip=0.1,
                          opt_func=opt_func, log_file=logResultFile)
        
        save_path = f'{model_dir}{program_name}/{selected_dataset_dir}/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)
        # # # 10. Evaluate the pruned model
        train_accuracy = 0.0
        test_accuracy = 0.0

        with open(outFile, 'a') as out_file:
            out_file.write(f'\n output of the {e}th iteration is written below\n')
            out_file.write(f'\n Train Accuracy: {train_accuracy}'
                           f'\n Test Accuracy  :  {test_accuracy} \n')
        out_file.close()

        save_path = f'{model_dir}{program_name}/selected/dataset_dir/vgg16_IntelIc_Prune_{e}_b_train'
        # save_path = f'/home3/pragnesh/Model/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)


# In[16]:
initialize_lists_for_pruning()
iterative_channel_pruning_saliency_block_wise(new_model_arg=new_model, 
    prune_module=module, block_list_l=block_list, prune_epochs=6)


# In[]
print()


