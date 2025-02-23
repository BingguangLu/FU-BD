# -*- coding: utf-8 -*-


import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim

import copy
import time
import numpy as np
import os
from sklearn.metrics import accuracy_score


import tqdm

def federated_learning(init_global_model, client_loaders, test_loader, FL_params):
    
    all_start_time = time.time()

    my_index = 3

    '''step0. Set up the total save folder for the training model, check it, and create it if it does not exist'''
    print(5*"#"+f" {my_index}.0 Initialize Federated models save path "+5*"#")
    FL_params.chosen_clients_str = ""
    for _ in FL_params.selected_clients:
        FL_params.chosen_clients_str += "_" + str(_)
    #FL_Model_dir = f"./{FL_params.model_result_name}/{FL_params.data_name}_clients{FL_params.chosen_clients_str}"
    FL_Model_dir = f"./{FL_params.model_result_name}/{FL_params.data_name}"
    FL_params.FL_Model_dir = FL_Model_dir
    
    if not os.path.exists(FL_Model_dir):
        os.makedirs(FL_Model_dir)
        print(5*"#"+f" {my_index}.0 Not found result dir, already made it: ",FL_Model_dir,5*"#")
    else:
        print(5*"#"+f" {my_index}.0 result dir is already existed: ",FL_Model_dir,5*"#")


    #FL_params.attack_clients_str = ""
    #for _ in FL_params.attack_clients_idx:
    #    FL_params.attack_clients_str += "_" + str(FL_params.selected_clients[_])

    

    '''step1. Train or import the initial federated learning model'''
    print(5*"#"+f" {my_index}.1 Federated Learning Start "+5*"#")
    FL_Primitive_Model_old_GMs_save_path = f'{FL_Model_dir}/FL_{FL_params.aggregation_method}_PGMs_Gp{FL_params.global_epoch}_Lp{FL_params.local_epoch}_bs{FL_params.local_batch_size}.pth'
    FL_Primitive_Model_old_CMs_save_path = f'{FL_Model_dir}/FL_{FL_params.aggregation_method}_PCMs_Gp{FL_params.global_epoch}_Lp{FL_params.local_epoch}_bs{FL_params.local_batch_size}.pth'
    std_time = time.time()
    if os.path.exists(FL_Primitive_Model_old_GMs_save_path) and os.path.exists(FL_Primitive_Model_old_CMs_save_path) and not FL_params.if_train:
        # Extract parameter
        old_GMs_dicts = torch.load(FL_Primitive_Model_old_GMs_save_path)
        old_CMs_dicts = torch.load(FL_Primitive_Model_old_CMs_save_path)
        # Initialization model
        old_GMs = [copy.deepcopy(init_global_model) for _ in range(len(old_GMs_dicts))]
        old_CMs = [copy.deepcopy(init_global_model) for _ in range(len(old_CMs_dicts))]
        # Load parameter
        for GMs_idx in range(len(old_GMs_dicts)):
            old_GMs[GMs_idx].load_state_dict(old_GMs_dicts[GMs_idx])
        for CMs_idx in range(len(old_CMs_dicts)):
            old_CMs[CMs_idx].load_state_dict(old_CMs_dicts[CMs_idx])

        old_epoch_result_in_test = np.load(FL_Primitive_Model_old_GMs_save_path.replace(".pth","_Rtest.npy"),allow_pickle=True).tolist()

        print(5*"#"+f" {my_index}.1 Find FL_Primitive_Model old_GMS & old_CMs, Load Successfuly! "+5*"#")
        
    else:
        # Training FL model
        old_GMs, old_CMs,  old_epoch_result_in_test = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
        # Extract parameter
        old_GMs_dicts = [model_temp.state_dict() for model_temp in old_GMs]
        old_CMs_dicts = [model_temp.state_dict() for model_temp in old_CMs]
        # Save parameter
        np.save(FL_Primitive_Model_old_GMs_save_path.replace(".pth","_Rtest.npy"),np.array(old_epoch_result_in_test,dtype=object))
        torch.save(old_GMs_dicts, FL_Primitive_Model_old_GMs_save_path)
        torch.save(old_CMs_dicts, FL_Primitive_Model_old_CMs_save_path)
        
        print(5*"#"+f" {my_index}.1 Not Find FL_Primitive_Model old_GMS & old_CMs, Train and Save Successfuly! "+5*"#")
    
    end_time = time.time()
    time_learn = ( end_time - std_time )

    

    '''step4. Prepare the output'''
    train_str = f"{FL_params.data_name}_clients{FL_params.chosen_clients_str}_{FL_params.aggregation_method}_Gepoch{FL_params.global_epoch}_Lepoch{FL_params.local_epoch}_batchsize{FL_params.local_batch_size}"
    result = ( (train_str, old_epoch_result_in_test[-1], len(old_epoch_result_in_test)))
    # print the total time spent
    print(5*"#"+f" Idx.{my_index} cost time: "+f"{time.time()-all_start_time}"+5*"#")
    
    
    #print(result)


    return old_GMs, old_CMs, result


def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):
    all_methods = ["fedavg", "median", "trim-mean"]
    # Initialize the model
    all_global_models = list()
    all_client_models = list()
    global_model = init_global_model

    all_global_models.append(copy.deepcopy(global_model))
    
    epoch_result_in_test = list()
    #epoch_result_in_target = list()
    epoch_result_in_test.append( [0,test(all_global_models[-1], test_loader)] )
    #epoch_result_in_target.append( [0,test_target(all_global_models[-1], test_loader, FL_params.target_idx)] )

    print('FL_Train parameters: Clients = {} ,Global epoch = {}, Local epoch = {}'.format(FL_params.chosen_clients_str[1:],FL_params.global_epoch,FL_params.local_epoch))
    desc_str = f"{FL_params.aggregation_method}|FL Primitive Model Training:"
    # Training model
    global_epoch_bar = tqdm.trange(FL_params.global_epoch,colour="yellow",desc=desc_str)
    for epoch in global_epoch_bar:
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)

        all_client_models += client_models
        if FL_params.aggregation_method == "fedavg":
            global_model = fedavg(client_models)
        elif FL_params.aggregation_method == "median":
            global_model = median(client_models)
        elif FL_params.aggregation_method == "trim-mean":
            global_model = trim_mean(client_models, FL_params.percentage)
        else:
            raise ValueError(f'No such method in aggregation, please check it! Only {all_methods}')
    
        all_global_models.append(copy.deepcopy(global_model))

        if FL_params.train_with_test:
            global_epoch_bar.set_postfix(A_L='{:.2e}'.format(epoch_result_in_test[-1][1][0]),# the loss of test set
                                        A_a='{:.1f}'.format(epoch_result_in_test[-1][1][1]),)# the accuracy of test set
            
            epoch_result_in_test.append( [epoch+1,test(all_global_models[-1], test_loader)] )
            #epoch_result_in_target.append( [epoch+1,test_target(all_global_models[-1], test_loader, FL_params.target_idx)] )

            
    return all_global_models, all_client_models, epoch_result_in_test
        
        

"""
Function:
For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
NOTE:The global model inputed is the global model for the previous round
    The output client_Models is the model that each user trained separately.
"""
#training sub function    
def global_train_once(global_model, client_data_loaders, test_loader, FL_params):

    # Device initialization
    device = FL_params.device
    device_cpu = torch.device("cpu")
    
    # Model and optimizer initialization
    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))
        

    # Single client is trained
    for client_idx in range(FL_params.N_client):

        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]
        
        model.to(device)
        model.train()
        
        #local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()
                
        
        model.to(device_cpu)
        client_models[client_idx] = model

    return client_models


"""
Function:
Test the performance of the model on the test set
"""
def test(model, test_loader, print_indicate=False):
    model.eval()
    model_device = (next(model.parameters())).device

    all_data = []
    all_target = []
    with torch.no_grad():
        for data, target in test_loader:
            all_data += data
            all_target += target
    
    all_data = torch.stack(all_data, dim=0).to(model_device)
    all_target = torch.stack(all_target, dim=0).to(model_device)
    

    output = model(all_data)
    criteria = nn.CrossEntropyLoss()
    test_loss = criteria(output, all_target) # sum up batch loss
    
    pred = torch.argmax(output,axis=1)
    test_acc = accuracy_score(pred.cpu(),all_target.cpu())*100
    test_loss /= len(test_loader.dataset)
    if print_indicate:
        print('Test set: Average loss: {:.8f}'.format(test_loss))         
        print('Test set: Average acc:  {:.4f}'.format(test_acc))    
    return (test_loss.item(), test_acc)


def test_target(model, test_loader, target_idx):
    model.eval()
    model_device = (next(model.parameters())).device

    all_data = []
    all_target = []
    with torch.no_grad():
        if type(target_idx) is not list:
            all_data.append(test_loader.dataset[target_idx][0])
            all_target.append(test_loader.dataset[target_idx][1])
        else:
            for idx in target_idx:
                all_data.append(test_loader.dataset[idx][0])
                all_target.append(test_loader.dataset[idx][1])
    
    all_data = torch.stack(all_data, dim=0).to(model_device)
    all_target = torch.tensor(all_target).to(model_device)
    

    output = model(all_data)
    criteria = nn.CrossEntropyLoss()
    test_loss = criteria(output, all_target) # sum up batch loss
    
    pred = torch.argmax(output,axis=1)
    test_acc = accuracy_score(pred.cpu(),all_target.cpu())*100
    test_loss /= len(test_loader.dataset)

    return (test_loss.item(), test_acc, output.cpu().tolist())


"""
Function:
FedAvg
"""    
def fedavg(local_models):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.

    Returns
    -------
    update_global_model
        Updated global model using fedavg algorithm
    """

    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()
    
    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())
    
    
    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0 
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        if "num_batches_tracked" in layer:
            avg_state_dict[layer] = local_state_dicts[client_idx][layer]
        else:
            avg_state_dict[layer] = avg_state_dict[layer] / len(local_models)
    
    global_model.load_state_dict(avg_state_dict)
    return global_model 

"""
Function:
Median
"""   
def median(local_models):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.

    Returns
    -------
    update_global_model
        Updated global model using fedavg algorithm
    """
    global_model = copy.deepcopy(local_models[0])
    median_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in median_state_dict.keys():
        weights = list()
        for client_idx in range(len(local_models)):
            weights.append(local_state_dicts[client_idx][layer])
        stacked_weights = torch.stack(weights)
        median_weight = torch.median(stacked_weights, dim=0).values
        median_state_dict[layer] = median_weight
    
    global_model.load_state_dict(median_state_dict)
    return global_model

"""
Function:
Trim-Mean
"""   
def trim_mean(local_models, percentage):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
    percentage : float
        DESCRIPTION.The parament of trim-mean algorithm. The value ranges from 0 to 0.5
    Returns
    -------
    update_global_model
        Updated global model using fedavg algorithm
    """
    global_model = copy.deepcopy(local_models[0])
    mean_state_dict = global_model.state_dict()

    if (percentage >= 0.5):
        raise ValueError('The percentage need to be smaller than 0.5')

    model_len = len(local_models)
    remove_num = int(model_len * percentage)
    end_index = remove_num * -1

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in mean_state_dict.keys():
        

        weights = list()
        for client_idx in range(len(local_models)):
            weights.append(local_state_dicts[client_idx][layer])
        stacked_weights = torch.stack(weights)

        sorted_weights, _ = torch.sort(stacked_weights, dim=0)

        trimmed_weights = sorted_weights[remove_num:end_index]

        mean_weight = trimmed_weights.mean(dim=0)

        mean_state_dict[layer] = mean_weight
    
    global_model.load_state_dict(mean_state_dict)

            
    return global_model