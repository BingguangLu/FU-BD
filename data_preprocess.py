# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

"""Function: load data"""
def data_init(FL_params):
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name,FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # Build a test data loader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # Evenly distribute the data into N-client copies according to the training trainset, and save all the segmented datasets in a list
    split_index = [int(trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)
    
    client_loaders = []

    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))

        '''
        By now, we have separated the local data of the client user and stored it in client_loaders.
        Each corresponds to a user's private data
        '''
    
    return client_loaders, test_loader


def data_init_non_iid(FL_params):
    import matplotlib.pyplot as plt
    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name,FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # Build a test data loader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # non-iid split
    classes = trainset.classes
    n_classes = len(classes)
    n_clients=FL_params.N_total_client

    labels = np.concatenate(
        [np.array(trainset.targets), np.array(testset.targets)], axis=0)


    client_sample_nums = balance_split(n_clients, len(trainset))
    client_idcs_dict = client_inner_dirichlet_partition(trainset.targets, n_clients, n_classes, dir_alpha=0.35, client_sample_nums=client_sample_nums, verbose=False)
    
    client_idcs = []
    for _ in client_idcs_dict.values():
        client_idcs.append(_)


    client_dataset = [torch.utils.data.Subset(trainset, indices) for indices in client_idcs]

    # This section describes how different labels are assigned to different clients
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_clients + 1.5, 1),
                label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                        c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(f"{FL_params.data_name}_noniid_plot.png")

    client_loaders = []

    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
    
        '''
        By now, we have separated the local data of the client user and stored it in client_loaders.
        Each corresponds to a user's private data
        '''
    
    return client_loaders, test_loader


def balance_split(num_clients, num_samples):
    """Assign same sample sample for each client.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    return client_sample_nums


def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha,
                                    client_sample_nums, verbose=True):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                    size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                    range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict


def data_set(data_name,transform_index=False):
    if not data_name in ['mnist','purchase','adult','cifar10']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')
    
    #model: 2 conv. layers followed by 2 FC layers
    if(data_name == 'mnist'):
        trainset = datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        testset = datasets.MNIST('./data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        
    #model: Similar to vgg network
    elif(data_name == 'cifar10'):

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # converting images to tensor
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
        # if the image dataset is black and white image, there can be just one number. 
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])


        trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)


        testset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    
    #model: 2 FC layers
    elif(data_name == 'purchase'):
        xx = np.load("./data/purchase/purchase_xx.npy")
        yy = np.load("./data/purchase/purchase_y2.npy")
        # yy = yy.reshape(-1,1)
        # enc = preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(yy)
        # yy = enc.transform(yy).toarray()
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
        
        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)
        
        trainset = TensorDataset(X_train_tensor,y_train_tensor)
        testset = TensorDataset(X_test_tensor,y_test_tensor)
        
        trainset.classes = ["0","1"]
        testset.classes = ["0","1"]

        trainset.targets = y_train_tensor
        testset.targets = y_test_tensor

        
    return trainset, testset


'''
def data_init_non_iid_client(FL_params):
    import matplotlib.pyplot as plt
    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name, FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # Build a test data loader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # non-iid split
    classes = trainset.classes
    n_classes = len(classes)
    n_clients = FL_params.N_total_client

    # 1. 提取标签为 0 的所有索引（client0 独享）
    client0_class0_indices = [i for i, target in enumerate(trainset.targets) if target == 0]

    # 2. 提取其他标签（非0）的索引
    other_indices = [i for i, target in enumerate(trainset.targets) if target != 0]

    # 3. 随机打乱其他标签的索引顺序
    np.random.shuffle(other_indices)

    # 4. 将其他数据均匀分配给所有客户端（包括 client0）
    other_client_indices = np.array_split(other_indices, n_clients)

    # 5. 构造每个客户端的样本索引列表：
    #    client0 同时拥有 class0 的数据和分得的其他数据，
    #    其他客户端仅获得分得的其他数据
    client0_indices = client0_class0_indices + list(other_client_indices[0])
    client_idcs = [client0_indices] + [list(idcs) for idcs in other_client_indices[1:]]

    # 构造各客户端数据集
    client_dataset = [torch.utils.data.Subset(trainset, indices) for indices in client_idcs]
    
    # 可视化各客户端标签分布
    labels = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)
    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" % c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(f"{FL_params.data_name}_noniid_plot.png")

    client_loaders = []
    for ii in range(n_clients):
        client_loaders.append(DataLoader(client_dataset[ii], 64, shuffle=True, **kwargs))
    
    return client_loaders, test_loader
'''

def data_init_non_iid_client(FL_params):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name, FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # Build a test data loader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # non-iid split
    classes = trainset.classes
    n_classes = len(classes)
    n_clients = FL_params.N_total_client

    # 1. 提取所有标签为 0 的索引，并随机打乱后只保留 1/10
    all_class0_indices = [i for i, target in enumerate(trainset.targets) if target == 0]
    np.random.shuffle(all_class0_indices)
    downsampled_class0_indices = all_class0_indices[:len(all_class0_indices) // 10]

    # 2. 提取其他标签（非 0）的索引，并随机打乱
    other_indices = [i for i, target in enumerate(trainset.targets) if target != 0]
    np.random.shuffle(other_indices)

    # 3. 将其他数据均匀分配给所有客户端（包括 client0）
    other_client_indices = np.array_split(other_indices, n_clients)

    # 4. 构造每个客户端的样本索引列表：
    #    client0 同时拥有降采样后的 class0 数据和分得的其他数据，
    #    其他客户端仅获得分得的其他数据
    client0_indices = list(downsampled_class0_indices) + list(other_client_indices[0])
    client_idcs = [client0_indices] + [list(idcs) for idcs in other_client_indices[1:]]

    # 构造各客户端数据集
    client_dataset = [torch.utils.data.Subset(trainset, indices) for indices in client_idcs]

    # 可视化各客户端标签分布
    labels = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, indices in enumerate(client_idcs):
        for idx in indices:
            label_distribution[labels[idx]].append(c_id)
    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" % c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(f"{FL_params.data_name}_noniid_plot.png")

    client_loaders = []
    for dataset in client_dataset:
        client_loaders.append(DataLoader(dataset, FL_params.local_batch_size, shuffle=True, **kwargs))
    
    return client_loaders, test_loader



'''
def data_init_non_iid_poison(FL_params):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    import matplotlib.pyplot as plt
    from PIL import Image  # 如果需要处理PIL图片
    # 根据设备选择 DataLoader 参数
    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}

    # 加载数据集
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name, FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # non-iid 划分
    classes = trainset.classes
    n_classes = len(classes)
    n_clients = FL_params.N_total_client

    # 1. client0获得所有标签为0的样本索引
    client0_indices = [i for i, target in enumerate(trainset.targets) if target == 0]
    # 2. 其他客户端获得其他标签的样本索引
    other_indices = [i for i, target in enumerate(trainset.targets) if target != 0]
    # 3. 随机打乱其他样本索引
    np.random.shuffle(other_indices)
    # 4. 将剩余样本均分给其他客户端
    other_client_indices = np.array_split(other_indices, n_clients - 1)
    # 5. 整体客户端索引列表
    client_idcs = [client0_indices] + list(other_client_indices)

    # 定义后门触发器函数（假设数据为tensor，形状为 (C, H, W) 且像素值在 [0, 1]）
    def add_backdoor_trigger(image):
        # 这里在右下角添加一个 3x3 的白色小块作为触发器
        img = image.clone()  # 防止直接修改原图
        img[:, -3:, -3:] = 1.0  # 将最后3行3列置为1（白色）
        return img

    # 定义一个包装数据集，在__getitem__中加入触发器（如果需要修改标签可传入poison_label）
    class PoisonedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, trigger_func, poison_label=None):
            self.dataset = dataset
            self.indices = indices
            self.trigger_func = trigger_func
            self.poison_label = poison_label

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data, label = self.dataset[self.indices[idx]]
            # 对数据加入后门触发器
            data = self.trigger_func(data)
            # 如果需要修改标签（例如所有被投毒样本统一改为目标标签），则可以设置poison_label
            if self.poison_label is not None:
                label = self.poison_label
            return data, label

    # 构造各个客户端的数据集
    client_dataset = []
    # 对于client 0，使用投毒数据集
    poisoned_client0 = PoisonedDataset(trainset, client0_indices, add_backdoor_trigger)
    client_dataset.append(poisoned_client0)
    # 其他客户端使用普通的子集
    for indices in other_client_indices:
        client_dataset.append(Subset(trainset, indices))

    # 可选：绘制各客户端标签分布（注意此处标签分布仍基于原标签）
    labels = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), [f"Client {c_id}" for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(f"{FL_params.data_name}_noniid_plot.png")

    # 构造各客户端数据加载器
    client_loaders = []
    for ds in client_dataset:
        client_loaders.append(DataLoader(ds, 64, shuffle=True, **kwargs))

    return client_loaders, test_loader
'''

def data_init_non_iid_poison(FL_params):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    # 定义后门触发函数：假设图像是 tensor，添加一个右下角 3x3 的白色块
    def add_backdoor_trigger(image):
        # 复制图像以避免修改原图
        image = image.clone()
        image[:, -3:, -3:] = 1.0
        return image

    # 自定义数据集：复制后门样本（选取非 class3 的样本），并将复制样本的标签统一改为 3
    class BackdoorSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, backdoor_indices):
            """
            dataset: 原始数据集
            indices: 客户端原始样本索引（干净样本）
            backdoor_indices: 后门候选样本索引（用于复制生成后门样本，条件为标签 != 3）
            """
            self.dataset = dataset
            self.indices = indices
            self.backdoor_indices = list(backdoor_indices)

        def __len__(self):
            # 数据集长度 = 干净样本数量 + 复制的后门样本数量
            return len(self.indices) + len(self.backdoor_indices)

        def __getitem__(self, idx):
            # 前 len(self.indices) 个样本为干净样本
            if idx < len(self.indices):
                global_idx = self.indices[idx]
                img, target = self.dataset[global_idx]
                return img, target
            else:
                # 剩余部分为复制的后门样本
                dup_idx = idx - len(self.indices)
                global_idx = self.backdoor_indices[dup_idx]
                img, _ = self.dataset[global_idx]
                img = add_backdoor_trigger(img)
                # 后门样本标签全部改为 3
                new_label = 3
                return img, new_label

    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name, FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # 构建测试集 DataLoader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # non-iid 数据划分
    classes = trainset.classes
    n_classes = len(classes)
    n_clients = FL_params.N_total_client

    labels = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)

    client_sample_nums = balance_split(n_clients, len(trainset))
    client_idcs_dict = client_inner_dirichlet_partition(trainset.targets, n_clients, n_classes, 
                                                        dir_alpha=0.4, client_sample_nums=client_sample_nums, verbose=False)
    
    client_idcs = []
    for idcs in client_idcs_dict.values():
        client_idcs.append(idcs)

    # 构造每个客户端的数据集（均为 torch.utils.data.Subset）
    client_dataset = [torch.utils.data.Subset(trainset, indices) for indices in client_idcs]

    # --- 针对后门数据处理 ---
    # 选定 client5 作为后门注入客户端（使用 client_idcs[5]）
    client5_indices = client_idcs[5]
    # 在该客户端中，选择所有非 class3 的样本作为后门候选样本
    backdoor_candidates = [idx for idx in client5_indices if trainset.targets[idx] != 3]
    np.random.shuffle(backdoor_candidates)
    # 复制所有候选样本，生成后门样本，并将后门标签统一改为 3
    backdoor_indices = backdoor_candidates
    client_dataset[5] = BackdoorSubset(trainset, client5_indices, backdoor_indices)
    # --- 结束后门数据处理 ---

    # 可视化各客户端标签分布（包括复制的后门样本）
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, dataset in enumerate(client_dataset):
        for i in range(len(dataset)):
            _, label = dataset[i]
            label_distribution[label].append(c_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" % c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients (Including Backdoor Samples)")
    plt.savefig(f"{FL_params.data_name}_noniid_plot.png")

    client_loaders = []
    for ii in range(n_clients):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
    
    return client_loaders, test_loader


