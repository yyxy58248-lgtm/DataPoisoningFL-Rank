from loguru import logger
import pathlib
import os
from federated_learning.arguments import Arguments
from federated_learning.datasets import CIFAR10Dataset
from federated_learning.datasets import FashionMNISTDataset
from federated_learning.datasets import MNISTDataset
from federated_learning.datasets import QMNISTDataset
from federated_learning.utils import generate_noniid_train_loader
from federated_learning.utils import generate_train_loader
from federated_learning.utils import generate_test_loader
from federated_learning.utils import save_data_loader_to_file
from federated_learning.datasets import MedMNISTDataset


if __name__ == '__main__':
    args = Arguments(logger)

    # # ---------------------------------
    # # ------------ CIFAR10 ------------
    # # ---------------------------------
    # dataset = CIFAR10Dataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/test_data_loader.pickle"

    # if not os.path.exists("data_loaders/cifar10"):
    #     pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)

    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)

    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)

    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # # ---------------------------------
    # # --------- Fashion-MNIST ---------
    # # ---------------------------------
    # dataset = FashionMNISTDataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/test_data_loader.pickle"

    # if not os.path.exists("data_loaders/fashion-mnist"):
    #     pathlib.Path("data_loaders/fashion-mnist").mkdir(parents=True, exist_ok=True)

    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)

    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)

    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # # ---------------------------------
    # # ------- Handwriting-MNIST -------
    # # ---------------------------------
    # dataset = MNISTDataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/mnist/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/mnist/test_data_loader.pickle"

    # if not os.path.exists("data_loaders/mnist"):
    #     pathlib.Path("data_loaders/mnist").mkdir(parents=True, exist_ok=True)

    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)

    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)

    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # # ---------------------------------
    # # ------- Writer-QMNIST -----------
    # # ---------------------------------
    # dataset = QMNISTDataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/qmnist/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/qmnist/test_data_loader.pickle"

    # if not os.path.exists("data_loaders/qmnist"):
    #     pathlib.Path("data_loaders/qmnist").mkdir(parents=True, exist_ok=True)

    # train_data_loader = generate_noniid_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)

    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)

    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)


    # ---------------------------------
    # ------- MedMNIST (DermaMNIST) -------
    # ---------------------------------
    print("正在生成 MedMNIST (DermaMNIST) 数据加载器...")
    print("  输入尺寸: 28x28x3 (RGB)")
    print("  输出类别: 7")
    
    dataset = MedMNISTDataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/medmnist/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/medmnist/test_data_loader.pickle"

    if not os.path.exists("data_loaders/medmnist"):
        pathlib.Path("data_loaders/medmnist").mkdir(parents=True, exist_ok=True)

    # 使用标准的IID训练数据分配（与论文中IID场景一致）
    # 注意：这里使用 generate_train_loader 而不是 generate_noniid_train_loader
    # 因为您复现的是4.2节的IID场景
    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)

    print("✅ MedMNIST (DermaMNIST) 数据生成完成！")
    print(f"   - 训练数据: {TRAIN_DATA_LOADER_FILE_PATH}")
    print(f"   - 测试数据: {TEST_DATA_LOADER_FILE_PATH}")
    
    # 打印一些统计信息
    print("\n数据集统计:")
    print(f"  训练样本数: 通过DataLoader加载")
    print(f"  测试样本数: 通过DataLoader加载")
    print(f"  图像尺寸: {dataset.train_data[0][0].shape if hasattr(dataset, 'train_data') else '未知'}")
