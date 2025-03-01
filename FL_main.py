
import torch
import numpy as np
import random


from model_initiation import mnist_model_init
from data_preprocess import data_init, data_init_non_iid, data_init_non_iid_client, data_init_non_iid_poison
from FL_base import federated_learning, test

# from Fed_Unlearn_base import federated_learning_unlearning
from options import args_parser


"""Step 0. Initialize Federated Unlearning parameters"""
class Arguments():
    def __init__(self, args):

        #Federated Learning Settings
        self.data_name = args.dataset
        self.data_split = args.data_split

        self.N_total_client = args.N_total_client
        self.N_client = args.N_client

        self.aggregation_method = args.aggregation  
        self.percentage = args.percentage        

        self.global_epoch = args.global_epoch  
        self.local_epoch = args.local_epoch

        self.selected_clients = np.random.choice(range(self.N_total_client),self.N_client,replace=False).tolist()
        
        

        #Model Training Settings
        self.local_batch_size = args.local_batch_size
        self.local_lr = args.local_learning_rate
        self.test_batch_size = args.local_batch_size

        
        # others
        self.device = args.device
        self.seed = args.seed
        self.model_result_name = args.result_dir
        self.train_with_test = True
        self.if_train = False
        self.if_unlearning_attack = False 
        # self.re_compute_influence = False





def Federated_Learning():

    """Step 1.Set the parameters for Federated Unlearning"""
    args = args_parser()
    FL_params = Arguments(args)
    print(args)
    seed = FL_params.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #torch.manual_seed(FL_params.seed)
    #kwargs for data loader 
    print(60*'=')
    print("Step1. Federated Learning Settings \n We use dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment.\n"))



    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    init_global_model = mnist_model_init(FL_params.data_name)

    if FL_params.data_split == 'iid':
        client_all_loaders, test_loader = data_init(FL_params)
    elif FL_params.data_split == 'noniid':
        client_all_loaders, test_loader = data_init_non_iid(FL_params)
    elif FL_params.data_split == 'client':
        client_all_loaders, test_loader = data_init_non_iid_client(FL_params)
    elif FL_params.data_split == 'poison':
        client_all_loaders, test_loader = data_init_non_iid_poison(FL_params)
    else:
        raise ValueError(f'No such data_split, please check it! Only iid or noniid')

    client_loaders = list()
    all_train_num = 0
    for idx in FL_params.selected_clients:
        client_loaders.append(client_all_loaders[idx])
        all_train_num += len(client_all_loaders[idx].dataset)
    


    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60*'=')
    print("Step3. Fedearated Learning Training...")

    old_GMs, old_CMs, result = federated_learning(init_global_model, 
                                                    client_loaders, 
                                                    test_loader, 
                                                    FL_params)
    
    print("print every global epoch test results:")
    for i in range(len(old_GMs)):
        print(60*'=')
        print(f"epoch: {i}")
        test(old_GMs[i], test_loader, True)
    
    print("All done!")

if __name__=='__main__':
    Federated_Learning()