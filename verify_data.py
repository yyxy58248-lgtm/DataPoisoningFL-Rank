import pickle
import torch
import numpy as np
from federated_learning.arguments import Arguments
from federated_learning.datasets import MedMNISTDataset
from loguru import logger

# 验证保存的数据加载器
print("="*50)
print("验证数据加载器")
print("="*50)

# 加载训练数据加载器
with open('data_loaders/medmnist/train_data_loader.pickle', 'rb') as f:
    train_loader = pickle.load(f)

# 加载测试数据加载器
with open('data_loaders/medmnist/test_data_loader.pickle', 'rb') as f:
    test_loader = pickle.load(f)

# 获取一个batch
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"\n训练数据 Batch {batch_idx}:")
    print(f"  图像形状: {data.shape}")
    print(f"  标签形状: {target.shape}")
    print(f"  图像数据类型: {data.dtype}")
    print(f"  图像值范围: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  标签示例: {target[:5]}")
    break

for batch_idx, (data, target) in enumerate(test_loader):
    print(f"\n测试数据 Batch {batch_idx}:")
    print(f"  图像形状: {data.shape}")
    print(f"  标签形状: {target.shape}")
    break

# 验证 MedMNISTDataset 直接加载
print("\n" + "="*50)
print("验证 MedMNISTDataset 直接加载")
print("="*50)

args = Arguments(logger)
dataset = MedMNISTDataset(args)
train_data = dataset.load_train_dataset()
test_data = dataset.load_test_dataset()

X, Y, Z = train_data
print(f"\n训练数据:")
print(f"  X shape: {X.shape}")
print(f"  Y shape: {Y.shape}")
print(f"  Z shape: {Z.shape} (用于Non-IID分组)")
print(f"  图像值范围: [{X.min():.3f}, {X.max():.3f}]")

X_test, Y_test = test_data
print(f"\n测试数据:")
print(f"  X_test shape: {X_test.shape}")
print(f"  Y_test shape: {Y_test.shape}")

print("\n✅ 数据验证完成！")
print(f"期望: 图像应为 28x28x3 的RGB图像")
print(f"实际: 训练图像 shape = {X.shape}")
if len(X.shape) == 4 and X.shape[2] == 28 and X.shape[3] == 28:
    print("✅ 图像尺寸正确: 28x28")
else:
    print(f"⚠️ 图像尺寸异常: {X.shape}")
