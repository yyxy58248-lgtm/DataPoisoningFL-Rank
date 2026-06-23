from federated_learning.arguments import Arguments
from federated_learning.nets import Cifar10CNN
from federated_learning.nets import FashionMNISTCNN
from federated_learning.nets import MNISTCNN
import os
import torch
from loguru import logger

if __name__ == '__main__':
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # # ---------------------------------
    # # ----------- Cifar10CNN ----------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN.model")
    # torch.save(Cifar10CNN().state_dict(), full_save_path)

    # # ---------------------------------
    # # -------- FashionMNISTCNN --------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN.model")
    # torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    # # ---------------------------------
    # # -------- MNISTCNN --------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "MNISTCNN.model")
    # torch.save(MNISTCNN().state_dict(), full_save_path)

    # # ---------------------------------
    # # -------- QMNISTCNN --------
    # # Note that we use the same model architecture for between QMNIST and MNIST
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "QMNISTCNN.model")
    # torch.save(MNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # -------- MedMNIST (DermaMNIST) -------
    # 使用原始CNN架构 (28x28输入，约42万参数)
    # ---------------------------------
    from federated_learning.nets import MedMNISTCNN
    
    full_save_path = os.path.join(args.get_default_model_folder_path(), "MedMNISTCNN.model")
    model = MedMNISTCNN(num_classes=7)
    torch.save(model.state_dict(), full_save_path)
    logger.info(f"✅ Saved MedMNIST CNN model to {full_save_path}")
    
    # 验证模型
    test_input = torch.randn(1, 3, 28, 28)
    test_output = model(test_input)
    logger.info(f"   Model test output shape: {test_output.shape} (expected: [1, 7])")
    logger.info(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✅ 默认模型生成完成！")
    print(f"   MedMNIST模型已保存到: {full_save_path}")
    print(f"   输入尺寸: 28x28x3")
    print(f"   输出类别: 7")
