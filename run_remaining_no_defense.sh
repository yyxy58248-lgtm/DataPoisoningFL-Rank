#!/bin/bash

# 创建目录
mkdir -p supplemental_experiments

# 定义需要运行的攻击
# GA: 只缺 nm=30
# ZG, SF, MS: 全部 nm

NM_VALUES=(5 10 15 20 25 30)

echo "=========================================="
echo "运行剩余的无防御实验"
echo "=========================================="
echo ""

TOTAL=0
# 计算总数
for attack in GA ZG SF MS; do
    if [ "$attack" = "GA" ]; then
        TOTAL=$((TOTAL + 1))  # 只缺 30
    else
        TOTAL=$((TOTAL + 6))
    fi
done
echo "总共需要运行: $TOTAL 个实验"
echo ""

CURRENT=0
SUCCESS=0
FAIL=0

# 1. GA - 只缺 nm=30
echo "[$((CURRENT+1))/$TOTAL] 运行: GA, nm=30, defense=None"
python3 gaussian_attack.py \
    --dataset medmnist \
    --def_method None \
    --p_workers 30 \
    --rep_n 0

if [ $? -eq 0 ]; then
    SUCCESS=$((SUCCESS + 1))
    echo "  ✅ 完成: GA, nm=30"
else
    FAIL=$((FAIL + 1))
    echo "  ❌ 失败: GA, nm=30"
fi
echo ""
CURRENT=$((CURRENT + 1))

# 2. ZG - 全部 6 个 nm
for nm in "${NM_VALUES[@]}"; do
    echo "[$((CURRENT+1))/$TOTAL] 运行: ZG, nm=$nm, defense=None"
    
    python3 zero_gradient_attack.py \
        --dataset medmnist \
        --def_method None \
        --p_workers "$nm" \
        --rep_n 0
    
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ 完成: ZG, nm=$nm"
    else
        FAIL=$((FAIL + 1))
        echo "  ❌ 失败: ZG, nm=$nm"
    fi
    echo ""
    CURRENT=$((CURRENT + 1))
done

# 3. SF - 全部 6 个 nm
for nm in "${NM_VALUES[@]}"; do
    echo "[$((CURRENT+1))/$TOTAL] 运行: SF, nm=$nm, defense=None"
    
    python3 sign_flipping_attack.py \
        --dataset medmnist \
        --def_method None \
        --p_workers "$nm" \
        --rep_n 0
    
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ 完成: SF, nm=$nm"
    else
        FAIL=$((FAIL + 1))
        echo "  ❌ 失败: SF, nm=$nm"
    fi
    echo ""
    CURRENT=$((CURRENT + 1))
done

# 4. MS - 全部 6 个 nm
for nm in "${NM_VALUES[@]}"; do
    echo "[$((CURRENT+1))/$TOTAL] 运行: MS, nm=$nm, defense=None"
    
    python3 shifted_mean_attack.py \
        --dataset medmnist \
        --def_method None \
        --p_workers "$nm" \
        --rep_n 0
    
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ 完成: MS, nm=$nm"
    else
        FAIL=$((FAIL + 1))
        echo "  ❌ 失败: MS, nm=$nm"
    fi
    echo ""
    CURRENT=$((CURRENT + 1))
done

echo "=========================================="
echo "全部实验完成！"
echo "=========================================="
echo "成功: $SUCCESS / $TOTAL"
echo "失败: $FAIL / $TOTAL"
echo ""
echo "验证生成的文件:"
echo "GA:  $(ls supplemental_experiments/gaussian_attack_None_*.log 2>/dev/null | wc -l) / 6"
echo "ZG:  $(ls supplemental_experiments/zero_gradient_attack_None_*.log 2>/dev/null | wc -l) / 6"
echo "SF:  $(ls supplemental_experiments/sign_flipping_attack_None_*.log 2>/dev/null | wc -l) / 6"
echo "MS:  $(ls supplemental_experiments/shifted_mean_attack_None_*.log 2>/dev/null | wc -l) / 6"
echo "=========================================="
