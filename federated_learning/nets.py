
class MedMNISTCNN(nn.Module):
    """用于MedMNIST（DermaMNIST）的CNN模型，输出7个类别"""
    
    def __init__(self):
        super(MedMNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 修改输出为7类（DermaMNIST有7种皮肤病变）
        self.fc = nn.Linear(7*7*32, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
