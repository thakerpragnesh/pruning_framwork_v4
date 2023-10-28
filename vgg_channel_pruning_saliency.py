
import torch  # Provides basic tensor operation and nn operation
import load_dataset as dl  # create dataloader for selected dataset
import load_model as lm  # facilitate loading and manipulating models
import train_model as tm  # Facilitate training of the model
import initialize_pruning as ip  # Initialize and provide basic parameter require for pruning
import facilitate_pruning as fp  # Compute Pruning Value and many things
import torch.nn.utils.prune as prune
import os  # use to access the files
from datetime import date


# In[3]: Ensure require directory for output is present.
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# ************************************ Initialize all the parameter ***************************************************
# In[2] Initialize all the string variable:String parameter for dataset
dataset_dir = '/home/pragnesh/project/Dataset/'
selected_dataset_dir = 'IntelIC'
train_folder = 'train'
test_folder = 'test'

# String Parameter for Model
loadModel = False
isTransferlearning = False
program_name = 'vgg_net_kernel_pruning_3Aug'
model_dir = '/home/pragnesh/project/Model/'
selectedModel = 'vgg16_IntelIc_Prune'
load_path = f'{model_dir}{program_name}/{selected_dataset_dir}/{selectedModel}'

# String parameter to Log Output
logDir = '/home/pragnesh/project/Logs/'
logResultFile = f'{logDir}{program_name}/{selected_dataset_dir}/result.log'
outFile = f'{logDir}{program_name}/{selected_dataset_dir}/lastResult.log'
outLogFile = f'{logDir}{program_name}/{selected_dataset_dir}/outLogFile.log'
if torch.cuda.is_available():
    device1 = torch.device('cuda')
else:
    device1 = torch.device('cpu')

# ************************ Ensure Output Directories are Present ******************************
# Ensure program name subdirectory is present inside model dir. ~/model_dir/program name/
ensure_dir(f'{model_dir}{program_name}/')
# Ensure selected dataset subdirectory is present inside program name dir ~/model_dir/program_name/selected_dataset
ensure_dir(f'{model_dir}{program_name}/{selected_dataset_dir}/')
# Ensure program name subdirectory is present inside Log Dir ~/logDir/program_name/
ensure_dir(f'{logDir}{program_name}')
# Ensure selected_dataset subdirectory is present inside program_name ~/logDir/program_name/selected_dataset
ensure_dir(f'{logDir}{program_name}/{selected_dataset_dir}/')
# *******************************************************************************************

# In[4]: Set the data loader to load the data for selected dataset
# set the image properties
dl.set_image_size(224)
dl.set_batch_size = 16
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir, selected_dataset_arg=selected_dataset_dir,
                             train_arg=train_folder, test_arg=test_folder)

# In[5]:load the saved model if we have any and if we don't load a standard model
if loadModel:  # Load the saved trained model
    new_model = torch.load(load_path, map_location=torch.device(device1))
else:  # Load the standard model from library
    # if we don't have any saved trained model download pretrained model for transfer learning
    new_model = lm.load_model(model_name='vgg16', number_of_class=6, pretrainval=isTransferlearning,
                              freeze_feature_arg=False, device_l=device1)

# In[6]: Change the print statement to write and store the intermediate result in selected file
today = date.today()
d1 = today.strftime("%d-%m")
print(f"\n...........OutLog For the {d1}................")
with open(outLogFile, 'a') as f:
    f.write(f"\n\n..........................OutLog For the {d1}......................\n\n")
f.close()
# ***************************************** Required Data List ********************************************************
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
# *********************************************************************************************************************


def initialize_pruning():
    global block_list  # ip.getBlockList('vgg16')
    global feature_list
    global conv_layer_index
    global module
    global prune_count
    global new_list
    global layer_number
    global st
    global en
    global candidate_conv_layer

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
                          f"\nEmpy candidate layer list = {candidate_conv_layer}"
                          )
        output_file.close()


def compute_conv_layer_saliency_channel_pruning(module_cand_conv, block_list_l, block_id, k=1):
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
            candidate_convolution_layer.append(fp.compute_saliency_score_channel(
                module_cand_conv[lno]._parameters['weight'],
                n=1,
                dim_to_keep=[0],
                k=prune_count[lno]))
        break
    return candidate_convolution_layer


''' Initialize pruning will initialize all the data structure required '''
initialize_pruning()
