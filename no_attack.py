from federated_learning.utils import no_noise
from federated_learning.utils.defense_methods import mandera_detect
from federated_learning.utils.defense_methods import multi_krum
from federated_learning.utils.defense_methods import bulyan
from federated_learning.utils.defense_methods import median
from federated_learning.utils.defense_methods import tr_mean
from federated_learning.utils.defense_methods import fltrust
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
import argparse
import numpy as np


# 定义一个不做任何修改的标签替换函数
def no_label_replacement(labels, unique_labels):
    """返回原始标签，不做任何修改"""
    return labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run No Attack Baseline')
    parser.add_argument('--p_workers', type=int,
                        help='number of poisoned workers (does nothing in no_attack)')
    parser.add_argument('--rep_n', type=int,
                        help='repetition number')
    parser.add_argument('--dataset', type=str,
                        help='which dataset?')
    parser.add_argument('--def_method', type=str,
                        help='which defense? (use "none" for no defense)')
    args = parser.parse_args()

    print(args)

    NUM_POISONED_WORKERS = args.p_workers
    NUM_OFFSET = args.rep_n

    DATASET = args.dataset
    
    if args.def_method == "none":
        START_EXP_IDX = 0  # 与 None 防御一致
        DEF_METHOD = None
    elif args.def_method == "mandera_detect":
        START_EXP_IDX = 100000
        DEF_METHOD = mandera_detect
    elif args.def_method == "multi_krum":
        START_EXP_IDX = 200000
        DEF_METHOD = multi_krum
    elif args.def_method == "bulyan":
        START_EXP_IDX = 300000
        DEF_METHOD = bulyan
    elif args.def_method == "median":
        START_EXP_IDX = 400000
        DEF_METHOD = median
    elif args.def_method == "tr_mean":
        START_EXP_IDX = 500000
        DEF_METHOD = tr_mean
    elif args.def_method == "fltrust":
        START_EXP_IDX = 600000
        DEF_METHOD = fltrust
    else:
        assert args.def_method in ["none", "mandera_detect", "multi_krum", "bulyan", "median", "tr_mean", "fltrust"]

    # 固定每次调用运行1个实验
    NUM_EXP = 1
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 100,
    }

    # 关键：使用空操作标签替换函数（不做任何修改）
    REPLACEMENT_METHOD = no_label_replacement
    NOISE_METHOD = no_noise

    # 根据数据集设置基础ID（与攻击实验保持一致）
    if DATASET == "FASHION":
        DATASET_BASE = 20000
    elif DATASET == "CIFAR10":
        DATASET_BASE = 60000
    elif DATASET == "MNIST":
        DATASET_BASE = 10020000
    elif DATASET == "medmnist":
        # 为medmnist分配独立范围，避免与MNIST冲突
        DATASET_BASE = 11000000
    else:
        assert DATASET in ["FASHION", "CIFAR10", "MNIST", "medmnist"]

    # 计算最终的实验ID
    # 格式：基础ID + 防御偏移 + 数据集偏移 + 恶意节点数*100 + 重复编号
    FINAL_START_IDX = START_EXP_IDX + DATASET_BASE + (NUM_POISONED_WORKERS * 100)

    for experiment_id in range(FINAL_START_IDX + NUM_OFFSET, FINAL_START_IDX + NUM_EXP + NUM_OFFSET):
        print(f"Running experiment with ID: {experiment_id}")
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id,
                noise_method=NOISE_METHOD, def_method=DEF_METHOD, dataset=DATASET)
