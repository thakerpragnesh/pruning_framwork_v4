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

today = date.today()
d1 = today.strftime("%d-%m")

# In[2]: String parameter for dataset
dataset_dir = '/home3/pragnesh/Dataset/';
selected_dataset_dir = 'IntelIC'
train_folder = 'train'
test_folder = 'test'

# In[3]: String Parameter for Model
loadModel = False
is_transfer_learning = False
program_name = 'vgg_net_kernel_pruning'
model_dir = '/home3/pragnesh/Model/'
selectedModel = 'vgg16_IntelIc_Prune'
# /home3/pragnesh/Model/vgg_net_kernel_pruning/IntelIC/vgg16_IntelIc_Prune
load_path = f'{model_dir}{program_name}/{selected_dataset_dir}/{selectedModel}'

# In[4]: String parameter to Log Output
logDir = '/home3/pragnesh/project/Logs/'
folder_path = f'{logDir}{program_name}/{selected_dataset_dir}/'
logResultFile = f'{folder_path}result.log'
outFile = f'{folder_path}lastResult.log'
outLogFile = f'{folder_path}outLogFile.log'


# In[5]: Check Cuda Devices
if torch.cuda.is_available():
    device1 = torch.device('cuda')
else:
    device1 = torch.device('cpu')

opt_func = torch.optim.Adam


# In[6]: Function to create folder if not exist
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[7]: Create output files if not present
ensure_dir(f'{model_dir}{program_name}/')
ensure_dir(f'{model_dir}{program_name}/{selected_dataset_dir}/')
ensure_dir(f'{logDir}{program_name}')
ensure_dir(f'{logDir}{program_name}/{selected_dataset_dir}/')

# In[8]: Set Image Properties
dl.set_image_size(224)
dl.set_batch_size = 16
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir,
                             selected_dataset_arg=selected_dataset_dir,
                             train_arg=train_folder, test_arg=test_folder)

# In[9]: Load appropriate model
if loadModel:  # Load the saved trained model
    new_model = torch.load(load_path, map_location=torch.device(device1))
else:  # Load the standard model from library
    new_model = lm.load_model(model_name='vgg16', number_of_class=6,
                              pretrainval=is_transfer_learning,
                              freeze_feature_arg=False, device_l=device1)

# In[11]: Create require lists for pruning
block_list = []; feature_list = []; conv_layer_index = []; module = []
prune_count = []; new_list = []; candidate_conv_layer = []
layer_number = 0; st = 0; en = 0


# In[12]: Initialize list with proper values
def initialize_lists_for_pruning():
    global block_list, feature_list, conv_layer_index, prune_count, module
    block_list = ip.create_block_list(new_model)  # ip.getBlockList('vgg16')
    feature_list = ip.create_feature_list(new_model)
    conv_layer_index = ip.find_conv_index(new_model)
    prune_count = ip.get_prune_count(module=module, blocks=block_list, max_pr=.1)
    module = ip.make_list_conv_param(new_model)


# In[13] Function to update the feature list after pruning
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


# In[ ]:
def compute_conv_layer_saliency_kernel_pruning(module_candidate_convolution, block_list_l, block_id, k=1):
    return module_candidate_convolution + block_list_l + block_id + k
    # replace the demo code above


# In[ ]:
class KernelPruningSaliency(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        return 0


# In[ ]:
def kernel_unstructured_saliency(kernel_module, name):
    KernelPruningSaliency.apply(kernel_module, name)
    return kernel_module

# In[ ]:




