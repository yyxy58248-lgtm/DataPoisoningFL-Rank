import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns

# ===================== 全局样式 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ===================== 核心配置 =====================
LOG_DIR = "../supplemental_experiments"

ATTACK_NAMES = {
    'GA': 'gaussian_attack',
    'ZG': 'zero_gradient_attack',
    'SF': 'sign_flipping_attack',
    'MS': 'shifted_mean_attack'
}
ATTACK_LABELS = ['GA', 'ZG', 'SF', 'MS']

DEFENSE_LIST = ['mandera_detect', 'multi_krum', 'bulyan', 'median', 'tr_mean', 'fltrust']
DEFENSE_LABELS = ['MANDERA', 'Krum', 'Bulyan', 'Median', 'Trim-mean', 'FLTrust']

NM_VALUES = [5, 10, 15, 20, 25, 30]


def parse_accuracy_from_log(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        acc_matches = re.findall(r'Test set: Accuracy: \d+/\d+ \((\d+)%\)', content)
        if acc_matches:
            return float(acc_matches[-1])
        return None
    except Exception:
        return None


def find_log_file(attack_name, defense, nm):
    if defense == 'none':
        fp = os.path.join(LOG_DIR, f"{attack_name}_none_{nm}.log")
        return fp if os.path.exists(fp) else None
    fp = os.path.join(LOG_DIR, f"{attack_name}_{defense}_{nm}.log")
    return fp if os.path.exists(fp) else None


print("=" * 60)
print("开始提取 Recovery Ratio 数据...")
print("=" * 60)

# ===================== 1. 提取 No Defense 准确率 =====================
no_defense_acc = {}
for attack in ATTACK_LABELS:
    no_defense_acc[attack] = {}
    for nm in NM_VALUES:
        log_file = find_log_file(ATTACK_NAMES[attack], 'none', nm)
        if log_file and os.path.exists(log_file):
            acc = parse_accuracy_from_log(log_file)
            no_defense_acc[attack][nm] = acc
            print(f"  No Defense - {attack} nm={nm}: {acc}%")
        else:
            no_defense_acc[attack][nm] = None
            print(f"  ⚠️ 未找到 {attack} nm={nm} 的 no_defense 日志")

# ===================== 2. 提取 Defense 准确率 =====================
defense_acc = {}
for defense in DEFENSE_LIST:
    defense_acc[defense] = {}
    for attack in ATTACK_LABELS:
        defense_acc[defense][attack] = {}
        for nm in NM_VALUES:
            log_file = find_log_file(ATTACK_NAMES[attack], defense, nm)
            if log_file and os.path.exists(log_file):
                acc = parse_accuracy_from_log(log_file)
                defense_acc[defense][attack][nm] = acc
            else:
                defense_acc[defense][attack][nm] = None

# ===================== 3. 计算 Baseline =====================
# 直接用 No Defense 作为 Baseline
# Recovery Ratio = (Acc_defense - Acc_no_defense) / Acc_no_defense
# 表示防御相对于无防御提升了百分之多少

ratio_data = {}
for defense in DEFENSE_LIST:
    ratio_data[defense] = {}
    for attack in ATTACK_LABELS:
        ratio_data[defense][attack] = {}
        for nm in NM_VALUES:
            acc_def = defense_acc[defense][attack].get(nm)
            acc_no = no_defense_acc[attack].get(nm)
            if acc_def is not None and acc_no is not None and acc_no > 0:
                # Recovery Ratio = (Defense - NoDefense) / NoDefense
                ratio = (acc_def - acc_no) / acc_no
                ratio_data[defense][attack][nm] = ratio
            else:
                ratio_data[defense][attack][nm] = None

# ===================== 4. 构建DataFrame =====================
x_labels = []
for attack in ATTACK_LABELS:
    for nm in NM_VALUES:
        x_labels.append(str(nm))

data_matrix = []
for defense in DEFENSE_LIST:
    row = []
    for attack in ATTACK_LABELS:
        for nm in NM_VALUES:
            ratio = ratio_data[defense][attack].get(nm)
            if ratio is not None:
                row.append(ratio)
            else:
                row.append(np.nan)
    data_matrix.append(row)

df_heatmap = pd.DataFrame(data_matrix, index=DEFENSE_LABELS, columns=x_labels)

print("\n" + "=" * 60)
print("Recovery Ratio 数据矩阵 (×100 = %):")
print(df_heatmap.round(4))
print("=" * 60)

# ===================== 5. 绘制热力图 =====================
print("\n绘制热力图...")

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5))

# 计算颜色范围（比例范围通常在 -1 到 10 之间，但可能更大）
max_val = max(abs(df_heatmap.min().min()), abs(df_heatmap.max().max()))
vmax = max(max_val, 0.5)

# 使用红-白-蓝配色
cmap = sns.diverging_palette(250, 10, as_cmap=True, s=90, l=45)

sns.heatmap(
    df_heatmap,
    annot=True,
    fmt='.2f',           # 显示两位小数
    cmap=cmap,
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    cbar_kws={
        'label': 'Recovery Ratio',
        'shrink': 0.8,
        'aspect': 30
    },
    annot_kws={'fontsize': 7}
)

ax.set_xlabel('Number of Malicious Nodes', fontsize=11, fontweight='bold')
ax.set_ylabel('Defense Method', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)

# 分组分割线
split_positions = [6, 12, 18]
for pos in split_positions:
    ax.axvline(x=pos, color='black', linewidth=2, alpha=0.5)

# 顶部标注攻击类型
attack_positions = [3, 9, 15, 21]
for i, attack in enumerate(ATTACK_LABELS):
    ax.text(attack_positions[i], -0.6, attack,
            ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax.transData, color='black')

plt.tight_layout()

plt.savefig('figures/recovery_ratio_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/recovery_ratio_heatmap.pdf', dpi=300, bbox_inches='tight')

print("\n" + "=" * 60)
print("✅ Recovery Ratio 热力图生成成功！")
print("   - PNG: figures/recovery_ratio_heatmap.png")
print("   - PDF: figures/recovery_ratio_heatmap.pdf")
print("=" * 60)

# ===================== 6. 打印统计 =====================
print("\n关键发现 (Recovery Ratio):")
print("-" * 50)
print("公式: Ratio = (Acc_defense - Acc_no_defense) / Acc_no_defense")
print("      Ratio = 0.5 表示防御比无防御提升了 50%")
print("-" * 50)

for defense in DEFENSE_LIST:
    ratios = [r for r in ratio_data[defense].values() for r in r.values() if r is not None]
    if ratios:
        avg_ratio = np.mean(ratios)
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        neg_count = sum(1 for r in ratios if r < 0)
        print(f"{defense:15s} | 平均: {avg_ratio:7.3f} | 最大: {max_ratio:7.3f} | 最小: {min_ratio:7.3f} | 负值: {neg_count:2d}/{len(ratios)}")

# ===================== 7. 特殊关注：FLTrust 的负值 =====================
print("\n" + "-" * 50)
print("FLTrust 负值详情 (Ratio < 0 表示防御比无防御更差):")
print("-" * 50)
for attack in ATTACK_LABELS:
    for nm in NM_VALUES:
        ratio = ratio_data['fltrust'][attack].get(nm)
        if ratio is not None and ratio < 0:
            acc_def = defense_acc['fltrust'][attack].get(nm)
            acc_no = no_defense_acc[attack].get(nm)
            print(f"  {attack} nm={nm}: Ratio = {ratio:.3f} (Defense={acc_def}%, NoDefense={acc_no}%)")