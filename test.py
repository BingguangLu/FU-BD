from model_initiation import mnist_model_init
from options import args_parser
from data_preprocess import data_init  # 根据你的数据划分方式选择合适的函数
#from evaluation import test
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np


def test_per_class(model, test_loader, print_indicate=False):
    """
    计算模型在干净测试数据上各类别的准确率。
    
    参数：
        model: 待测试模型
        test_loader: 测试数据的 DataLoader
        print_indicate: 是否打印每个类别的准确率
        
    返回：
        per_class_acc: 字典，键为类别标签，值为该类别的准确率（百分比）
    """
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    classes = np.unique(all_targets.numpy())
    per_class_acc = {}
    for c in classes:
        idx = (all_targets == c)
        if idx.sum().item() > 0:
            acc = (all_preds[idx] == all_targets[idx]).float().mean().item() * 100
        else:
            acc = 0
        per_class_acc[c] = acc
    
    if print_indicate:
        for c, acc in per_class_acc.items():
            print("Class {}: Accuracy: {:.2f}%".format(c, acc))
    
    return per_class_acc

def test_with_backdoor(model, test_loader, backdoor_target=3, print_indicate=False):
    """
    测试模型在干净数据（acc）和带后门触发器数据（bd ACR）上的表现。

    参数：
        model: 待测试的全局模型
        test_loader: 测试集 DataLoader
        backdoor_target: 后门攻击期望的目标标签（默认设为 0）
        print_indicate: 是否打印测试结果

    返回：
        avg_clean_loss: 干净测试数据的平均损失
        clean_acc: 干净数据上的准确率（百分比）
        bd_acr: 加入后门触发器后，模型输出为目标标签的比例（百分比）
    """
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    total_clean_loss = 0.0
    total_correct_clean = 0
    total_samples = 0

    total_bd_success = 0  # 后门攻击成功数量
    total_bd_samples = 0  # 测试样本总数

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            # --- 干净数据上的测试 ---
            output = model(data)
            loss = criterion(output, target)
            total_clean_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct_clean += (pred == target).sum().item()
            total_samples += data.size(0)

            # --- 加入后门触发器的测试 ---
            # 这里直接在 batch 数据上添加后门触发器：
            # 复制一份数据，将每张图的右下角3x3区域置为1（白色）
            triggered_data = data.clone()
            triggered_data[:, :, -3:, -3:] = 1.0
            bd_output = model(triggered_data)
            bd_pred = bd_output.argmax(dim=1)
            total_bd_success += (bd_pred == backdoor_target).sum().item()
            total_bd_samples += data.size(0)

    avg_clean_loss = total_clean_loss / total_samples
    clean_acc = total_correct_clean / total_samples * 100
    bd_acr = total_bd_success / total_bd_samples * 100

    if print_indicate:
        print('Clean Test set: Average loss: {:.8f}'.format(avg_clean_loss))
        print('Clean Test set: Accuracy: {:.2f}%'.format(clean_acc))
        print('Backdoor Attack Success Rate: {:.2f}%'.format(bd_acr))

    return avg_clean_loss, clean_acc, bd_acr


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

# 解析参数并构造 FL_params
args = args_parser()
FL_params = Arguments(args)

# 初始化测试集（以 iid 为例，noniid 或其它方式类似）
client_all_loaders, test_loader = data_init(FL_params)

# 初始化模型（确保使用和训练时一致的模型）
global_model = mnist_model_init(FL_params.data_name)

# 拼接文件保存路径
model_path = f"./{FL_params.model_result_name}/{FL_params.data_name}/FL_{FL_params.aggregation_method}_PGMs_Gp{FL_params.global_epoch}_Lp{FL_params.local_epoch}_bs{FL_params.local_batch_size}.pth"

# 加载保存的模型列表（注意：保存的是一个列表，每个元素是一个 state_dict）
saved_state_dicts = torch.load(model_path)

# 取出最后一个 state_dict，并加载到模型中
final_state_dict = saved_state_dicts[-1]
global_model.load_state_dict(final_state_dict)

# 对最终全局模型进行测试
per_class_acc = test_per_class(global_model, test_loader, print_indicate=True)
avg_clean_loss, clean_acc, bd_acr = test_with_backdoor(global_model, test_loader, print_indicate=True)
