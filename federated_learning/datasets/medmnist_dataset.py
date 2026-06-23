import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import medmnist
from medmnist import DermaMNIST
from loguru import logger
import os

class MedMNISTDataset(Dataset):
    """
    MedMNIST数据集包装器 - 使用28x28原始分辨率（论文标准尺寸）
    用于 IID 场景
    """
    def __init__(self, args, split='train', download=False):
        self.args = args
        self.split = split
        
        # 28x28图像的预处理（MEDMNIST原始尺寸）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 检查数据是否存在，如果不存在则下载
        data_path = './data/medmnist'
        if not os.path.exists(data_path):
            logger.info(f"Data not found at {data_path}, will download...")
            download = True
        
        # 使用原始尺寸 size=28（MEDMNIST默认）
        self.dataset = DermaMNIST(
            root='./data',
            split=split,
            download=download,
            transform=self.transform,
            size=28,
            as_rgb=True  # 使用RGB图像（3通道）
        )
        
        logger.info(f"Loaded 28x28 DermaMNIST {split} set: {len(self.dataset)} samples")
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        
        # 确保标签是整数标量
        if isinstance(label, np.ndarray):
            label = label.item()
        elif isinstance(label, torch.Tensor):
            label = label.item()
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)


# 为了兼容 medmnist.py 中的导入，保持类名一致
# 如果 medmnist.py 导入的是 BaseMedMNISTDataset，可以添加别名
BaseMedMNISTDataset = MedMNISTDataset


# 测试代码
if __name__ == '__main__':
    from federated_learning.arguments import Arguments
    from loguru import logger
    
    args = Arguments(logger)
    dataset = MedMNISTDataset(args, split='train', download=True)
    print(f"✅ Dataset created successfully!")
    print(f"   Number of samples: {len(dataset)}")
    print(f"   Image shape: {dataset[0][0].shape}")
    print(f"   Label: {dataset[0][1]}")
