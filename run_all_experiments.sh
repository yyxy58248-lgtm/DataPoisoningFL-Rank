#!/bin/bash

# 创建结果目录
mkdir -p experiment_results

# 定义要测试的防御方法
DEFENSES=("mandera_detect" "krum" "bulyan" "median" "trimmed_mean" "fltrust")

# 定义恶意节点数量
POISONED_WORKERS=(5 10 15 20 25 30)

# 定义攻击类型
ATTACKS=("gaussian_attack" "shifted_mean_attack" "sign_flipping_attack" "zero_gradient_attack")

# 记录开始时间
echo "=========================================" | tee experiment_results/experiment_log.txt
echo "开始批量实验: $(date)" | tee -a experiment_results/experiment_log.txt
echo "=========================================" | tee -a experiment_results/experiment_log.txt

# 循环运行所有组合
for attack in "${ATTACKS[@]}"; do
    for defense in "${DEFENSES[@]}"; do
        for workers in "${POISONED_WORKERS[@]}"; do
            echo "" | tee -a experiment_results/experiment_log.txt
            echo "运行: $attack | 防御: $defense | 恶意节点: $workers" | tee -a experiment_results/experiment_log.txt
            echo "----------------------------------------" | tee -a experiment_results/experiment_log.txt
            
            # 运行实验
            python ${attack}.py \
                --p_workers $workers \
                --rep_n 1 \
                --dataset medmnist \
                --def_method $defense \
                2>&1 | tee -a experiment_results/${attack}_${defense}_${workers}.log
            
            # 提取最终准确率
            grep "Test set: Accuracy:" experiment_results/${attack}_${defense}_${workers}.log | tail -1 >> experiment_results/accuracy_summary.txt
            
            echo "完成: $(date)" | tee -a experiment_results/experiment_log.txt
        done
    done
done

echo "=========================================" | tee -a experiment_results/experiment_log.txt
echo "所有实验完成: $(date)" | tee -a experiment_results/experiment_log.txt
echo "=========================================" | tee -a experiment_results/experiment_log.txt
