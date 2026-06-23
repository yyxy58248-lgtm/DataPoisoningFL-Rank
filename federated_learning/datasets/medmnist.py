import numpy as np
import torch
from torch.utils.data import DataLoader
from .medmnist_dataset import MedMNISTDataset as BaseMedMNISTDataset

try:
    from .dataset import Dataset
except:
    from dataset import Dataset

class MedMNISTDataset(Dataset):
    """
    MedMNIST数据集（继承框架Dataset类）
    用于与FL框架集成，支持Non-IID数据分配
    """
    
    def __init__(self, args):
        super(MedMNISTDataset, self).__init__(args)
    
    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading MedMNIST (DermaMNIST) train data")
        
        # 使用基础数据集类
        train_dataset = BaseMedMNISTDataset(
            self.get_args(), 
            split='train', 
            download=False
        )
        
        # 获取所有数据
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        batch_data = next(iter(train_loader))
        images, batch_labels = batch_data
        
        # 确保标签格式正确
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.numpy()
        if len(batch_labels.shape) > 1:
            batch_labels = batch_labels.flatten()
        
        # 重要：返回三个值 (images, labels, labels) 
        # 第三个值用标签作为分组信息，实现Non-IID分配（类似QMNIST的作者ID）
        train_data = (images.numpy(), batch_labels, batch_labels.copy())
        
        self.get_args().get_logger().debug(f"Finished loading MedMNIST train data: {len(train_dataset)} samples")
        
        return train_data
    
    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading MedMNIST (DermaMNIST) test data")
        
        # 使用基础数据集类
        test_dataset = BaseMedMNISTDataset(
            self.get_args(), 
            split='test', 
            download=False
        )
        
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        test_data = self.get_tuple_from_data_loader(test_loader)
        
        self.get_args().get_logger().debug(f"Finished loading MedMNIST test data: {len(test_dataset)} samples")
        
        return test_data


if __name__ == "__main__":
    from federated_learning.arguments import Arguments
    from loguru import logger
    
    args = Arguments(logger)
    dataset = MedMNISTDataset(args)
    
    print("Testing MedMNISTDataset...")
    train_data = dataset.load_train_dataset()
    test_data = dataset.load_test_dataset()
    
    X, Y, Z = train_data
    print(f"Train - X shape: {X.shape}")
    print(f"Train - Y shape: {Y.shape}")
    print(f"Train - Z shape: {Z.shape} (group IDs for Non-IID)")
    
    X_test, Y_test = test_data
    print(f"Test - X shape: {X_test.shape}")
    print(f"Test - Y shape: {Y_test.shape}")
    print("✅ MedMNISTDataset test passed!")
