import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 超参数设置
num_clients = 5        # 客户端数量
local_epochs = 1       # 每个客户端本地训练的轮数
batch_size = 32        # 批次大小
num_rounds = 20         # 联邦学习的通信轮数
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_classes = 10

# 数据预处理：将图片转换为 tensor 并归一化
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt

# 固定随机种子，保证结果可复现
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# 假设 train_dataset 已经定义好 (例如 MNIST)，并且 train_dataset.targets 是 Tensor 或 numpy 数组

# 1. 定义插入后门触发器的函数：在右下角插入 3×3 白色方块
def insert_backdoor_trigger(img):
    """
    在 28×28 灰度图像右下角插入 3×3 的白色方块。
    如果输入是 Tensor，则先转换为 PIL Image。
    输入: PIL Image 或 Tensor
    输出: 插入触发器后的 PIL Image
    """
    # 如果输入为 Tensor，则转换为 PIL Image
    if isinstance(img, torch.Tensor):
        img = ToPILImage()(img)
    # 将 PIL 图像转换为 Tensor
    img_tensor = ToTensor()(img)  # 形状: [1, 28, 28]
    # 在右下角填充 1.0 (即白色)
    img_tensor[0, 25:28, 25:28] = 1.0
    # 转回 PIL Image
    return ToPILImage()(img_tensor)

# 2. 自定义一个内存数据集，用于存储修改后的图像
class InMemoryDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: [(PIL_image, label), ...]
        transform: 后续训练用的 transforms (如标准化等)
        """
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_pil, label = self.data_list[idx]
        if self.transform:
            img_tensor = self.transform(img_pil)
            return img_tensor, label
        else:
            return img_pil, label

# 3. 从整个训练集中抽取 1000 个样本做后门，并保证所有类别都至少出现一次
def create_backdoor_dataset(train_dataset, n_backdoor=1000):
    # 获取标签数组
    targets_array = (train_dataset.targets.numpy() 
                     if isinstance(train_dataset.targets, torch.Tensor) 
                     else train_dataset.targets)
    n_classes = len(np.unique(targets_array))
    total_samples = len(train_dataset)
    
    # （1）先保证每个类别至少选 1 个
    selected_indices = []
    for cls in range(n_classes):
        class_indices = np.where(targets_array == cls)[0]
        # 随机选 1 个该类别样本
        idx = np.random.choice(class_indices)
        selected_indices.append(idx)
    
    # （2）剩余 n_backdoor - n_classes 个样本，从整个训练集中随机选
    needed = n_backdoor - n_classes
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    
    for idx in all_indices:
        if idx not in selected_indices:
            selected_indices.append(idx)
            if len(selected_indices) == n_backdoor:
                break
    
    # （3）对选出的样本插入后门触发器，强制标签为 0
    backdoor_data = []
    for idx in selected_indices:
        img, _ = train_dataset[idx]
        img_triggered = insert_backdoor_trigger(img)
        # 强制改为标签 0
        backdoor_data.append((img_triggered, 0))
    
    # （4）构建 InMemoryDataset 并返回
    # 如果你后续需要标准化，可以在 transform 中加上 Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    backdoor_dataset = InMemoryDataset(backdoor_data, transform=transform)
    return backdoor_dataset

# 4. 生成后门数据集并可视化检查
backdoor_dataset = create_backdoor_dataset(train_dataset, n_backdoor=1000)

print(f"后门数据集中样本数量: {len(backdoor_dataset)}")

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset
import matplotlib.pyplot as plt

# 固定随机种子（如有需要，可在需要随机操作的地方使用）
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# MNIST 有 10 个类别，假设我们想要将它们两两分配给 5 个客户端
client_class_map = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9]
}

# 1. 根据映射为每个客户端筛选对应类别的数据
client_subsets = []
for client_id, classes in client_class_map.items():
    # 如果 train_dataset.targets 是 Tensor，则先转换为 numpy
    targets = train_dataset.targets.numpy() if isinstance(train_dataset.targets, torch.Tensor) else train_dataset.targets
    
    # np.isin(targets, classes) 返回布尔数组，True 表示标签在 classes 中
    indices = np.where(np.isin(targets, classes))[0].tolist()
    
    subset = Subset(train_dataset, indices)
    client_subsets.append(subset)
    print(f"Client {client_id} 拥有类别 {classes}，数据量: {len(indices)}")

# 2. 可视化每个客户端中的类别分布
n_clients = len(client_subsets)
n_classes = 10  # MNIST 有 10 个类别
client_class_counts = []

# 遍历每个客户端的 Subset
for subset in client_subsets:
    # 当前客户端对应的数据索引
    indices = subset.indices
    # 只拿到该客户端的标签
    targets = train_dataset.targets.numpy() if isinstance(train_dataset.targets, torch.Tensor) else train_dataset.targets
    labels = targets[indices]
    
    # 统计每个类别出现的次数
    counts = [np.sum(labels == i) for i in range(n_classes)]
    client_class_counts.append(counts)

client_class_counts = np.array(client_class_counts)  # shape: (n_clients, n_classes)
client_subsets[0] = ConcatDataset([client_subsets[0], backdoor_dataset])
print("Client0 新的数据量:", len(client_subsets[0]))
client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_subsets]

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 将 28x28 的图像展平为 784 维向量
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def local_train(model, dataloader, epochs, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    # 返回训练后的模型参数
    return model.state_dict()


def fed_avg(global_model, client_state_dicts, client_data_counts):
    global_state_dict = global_model.state_dict()
    # 初始化全局模型参数为零
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])
    total_samples = sum(client_data_counts)
    # 对每个客户端的参数按数据量权重累加
    for client_state, n_samples in zip(client_state_dicts, client_data_counts):
        weight = n_samples / total_samples
        for key in global_state_dict.keys():
            global_state_dict[key] += client_state[key] * weight
    # 更新全局模型
    global_model.load_state_dict(global_state_dict)
    return global_model


import torch
from torchvision.transforms import ToPILImage, ToTensor

def evaluate_model(model, device, test_loader, n_classes=10, backdoor_target=0, backdoor_fn=insert_backdoor_trigger):
    """
    评估模型在正常测试集上的表现以及后门攻击成功率（ASR）。
    
    正常测试：
      - 计算总体准确率和每个类别的准确率。
      
    后门测试：
      - 对测试集中每个样本，先用 backdoor_fn 添加后门触发器，
        然后判断模型是否将其预测为 backdoor_target（例如 0）。
      - ASR = 后门样本中预测为目标标签的比例（百分比）。
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0 for _ in range(n_classes)]
    class_total = [0 for _ in range(n_classes)]
    
    # 用于后门测试
    backdoor_success = 0
    backdoor_total = 0
    
    # 遍历测试集（正常测试）
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
                    
            # 后门测试：对当前 batch 中的每个样本添加后门触发器后检测
            for j in range(data.size(0)):
                # 将 tensor 转为 PIL Image（先将该样本移回 CPU）
                pil_img = ToPILImage()(data[j].cpu())
                # 添加后门触发器
                triggered_img = backdoor_fn(pil_img)
                # 转为 tensor（注意后续可选归一化操作，这里仅用 ToTensor）
                triggered_tensor = ToTensor()(triggered_img).unsqueeze(0).to(device)
                output_bd = model(triggered_tensor)
                _, predicted_bd = torch.max(output_bd, 1)
                backdoor_total += 1
                if predicted_bd.item() == backdoor_target:
                    backdoor_success += 1
    
    overall_accuracy = 100.0 * correct / total
    class_accuracies = [100.0 * c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)]
    asr = 100.0 * backdoor_success / backdoor_total if backdoor_total > 0 else 0.0
    
    return overall_accuracy, class_accuracies, asr

# 计算每个客户端数据量
client_data_counts = [len(dataset) for dataset in client_subsets]
print("每个客户端数据量:", client_data_counts)

global_model = SimpleNN().to(device)


for r in range(num_rounds):
    print(f"==== 第 {r+1} 轮通信 ====")
    client_state_dicts = []
    local_models = []
    
    # 遍历每个客户端，进行本地训练和测试
    for c_id, client_loader in enumerate(client_loaders):
        local_model = SimpleNN().to(device)
        # 同步全局模型参数到客户端
        local_model.load_state_dict(global_model.state_dict())
        # 客户端本地训练
        local_state = local_train(local_model, client_loader, local_epochs, device)
        client_state_dicts.append(local_state)
        local_models.append(local_model)
        
        # 测试当前客户端本地模型在整个测试集上的表现
        overall_acc, class_acc, backdoor_asr = evaluate_model(local_model, device, test_loader, n_classes)
        print(f"Client {c_id} local model test accuracy: {overall_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {backdoor_asr:.2f}%")
        for i in range(n_classes):
            print(f"  Class {i} accuracy: {class_acc[i]:.2f}%")
    
    # 使用数据量加权的 FedAvg 聚合各客户端模型
    global_model = fed_avg(global_model, client_state_dicts, client_data_counts)
    
    # 选择性：测试聚合后的全局模型
    global_acc, global_class_acc, backdoor_asr = evaluate_model(global_model, device, test_loader, n_classes)
    print(f"Global model after round {r+1} test accuracy: {global_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {backdoor_asr:.2f}%")
    for i in range(n_classes):
        print(f"  Global model Class {i} accuracy: {global_class_acc[i]:.2f}%")
    print("\n")


# %% [markdown]
# ## Step 1: 从 test_loader 抽取20张图像并显示
#
# 直接从 test_loader 中抽取样本，不使用 test_dataset。
# 这里我们遍历 test_loader 的 batch，直到累积足够的样本，
# 然后将每个样本转换为 PIL Image 后显示。

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

# 设置抽取样本数
sample_size = 20
sample_images = []
sample_labels = []

# 从 test_loader 中抽取样本
for data, labels in test_loader:
    # data 的形状为 [batch_size, C, H, W]
    for img, label in zip(data, labels):
        # 将 tensor 转换为 PIL Image 进行显示
        pil_img = ToPILImage()(img)
        sample_images.append(pil_img)
        sample_labels.append(label.item())
        if len(sample_images) >= sample_size:
            break
    if len(sample_images) >= sample_size:
        break

# %% [markdown]
# ## Step 2: 用 global_model 对 Step 1 抽取的数据生成预测并显示结果
#
# 使用之前抽取的 20 张图像（变量 sample_images 和 sample_labels），生成预测结果，
# 并利用 pandas 表格显示每个样本的真实标签和预测标签，同时计算准确率。
# 如果图像已经是 Tensor，则先转换为 PIL Image 再进行预处理。

# %%
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import ToPILImage

pred_labels = []

global_model.eval()  # 确保模型处于评估模式

for i, img in enumerate(sample_images):
    # 如果图像是 Tensor，则转换为 PIL Image
    if isinstance(img, torch.Tensor):
        img = ToPILImage()(img)
    # 对图像进行预处理，转换为模型输入格式
    #img_tensor = eval_transform(img).unsqueeze(0).to(device)
    img_tensor = transform(img).to(device)
    with torch.no_grad():
        output = global_model(img_tensor)
        _, pred = torch.max(output, 1)
        #print(pred)
    pred_labels.append(pred.item())

# 构造结果表格
df = pd.DataFrame({
    "Sample Index": np.arange(len(sample_images)),
    "True Label": sample_labels,
    "Predicted Label": pred_labels
})

print(df)

# 计算准确率
accuracy = np.mean(np.array(sample_labels) == np.array(pred_labels)) * 100
print(f"\nAccuracy on the extracted {len(sample_images)} samples: {accuracy:.2f}%")
