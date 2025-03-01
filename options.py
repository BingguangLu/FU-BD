
import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser(description='Training a FL model and testing unlearning attack')    
    
    # Federated Learning Settings
    parser.add_argument('--dataset', type=str, default='mnist', help="dataset name")# purchase, cifar10, mnist
    parser.add_argument('--data_split', type=str, default='iid', choices=['iid','noniid', 'client', 'poison'], help="data set partitioning method")

    parser.add_argument('--N_total_client',type=int, default=10, help='the total number of clients in the data set')
    parser.add_argument('--N_client',type=int, default=10, help='the chosen number of clients for FL model training')

    parser.add_argument('--aggregation', type=str, default='trim-mean', choices=['fedavg','median','trim-mean'], help="aggregation method")
    parser.add_argument('--percentage', type=float, default=0.1, help='remove percentage of each side in trim-mean aggregation method')
    
    parser.add_argument('--global_epoch',type=int, default=20, help='the global epoch for FL model training')
    parser.add_argument('--local_epoch',type=int, default=10, help='the local epoch in each client for FL model training')

    
    # Model Training Settings
    parser.add_argument('--local_batch_size', type=int, default=64, help='the local batch size for each client')
    parser.add_argument('--local_learning_rate', type=float, default=0.005, help='the local learning rate for each client')
    

    # others
    parser.add_argument('--seed',type=int,default=1,help='random seed')
    parser.add_argument('--device',type=str,default="cuda:0" if torch.cuda.is_available() else "cpu",help='torch device')
    parser.add_argument('--result_dir',type=str,default="./result",help='result filename (includes path)')


    args = parser.parse_args()
    return args
