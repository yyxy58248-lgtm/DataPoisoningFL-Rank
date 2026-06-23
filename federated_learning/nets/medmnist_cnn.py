import torch
import torch.nn as nn
import torch.nn.functional as F

class MedMNISTCNN(nn.Module):
    """
    用于MEDMNIST (DERMAMNIST) 的简单CNN网络
    输入: 28x28x3 (DERMAMNIST是彩色RGB图像)
    输出: 7个类别
    """
    
    def __init__(self, num_classes=7):
        super(MedMNISTCNN, self).__init__()
        
        # 卷积层1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 卷积层2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 计算全连接层的输入维度
        # 输入28x28 -> 经过两次池化后变成7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 输入: batch_size x 3 x 28 x 28
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28 -> 14
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14 -> 7
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# 测试代码
if __name__ == '__main__':
    model = MedMNISTCNN(num_classes=7)
    
    # 测试28x28输入
    test_input_28 = torch.randn(2, 3, 28, 28)
    output_28 = model(test_input_28)
    print(f"✅ 原始CNN模型创建成功")
    print(f"   输入形状 (28x28): {test_input_28.shape}")
    print(f"   输出形状: {output_28.shape}")
    print(f"   期望输出: torch.Size([2, 7])")
    print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 验证输出正确
    assert output_28.shape == (2, 7), f"输出形状错误: {output_28.shape}"
    print(f"\n✅ 模型验证通过！")
    
    # 显示模型结构
    print(f"\n模型结构:")
    print(model)
