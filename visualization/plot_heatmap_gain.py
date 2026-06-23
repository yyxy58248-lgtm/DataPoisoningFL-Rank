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
print("开始提取 Recovery Gain 数据...")
print("=" * 60)

# 1. No Defense
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

# 2. Defense
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

# 3. Recovery Gain
recovery_data = {}
for defense in DEFENSE_LIST:
    recovery_data[defense] = {}
    for attack in ATTACK_LABELS:
        recovery_data[defense][attack] = {}
        for nm in NM_VALUES:
            acc_def = defense_acc[defense][attack].get(nm)
            acc_no = no_defense_acc[attack].get(nm)
            if acc_def is not None and acc_no is not None:
                recovery_data[defense][attack][nm] = acc_def - acc_no
            else:
                recovery_data[defense][attack][nm] = None

# ===================== 4. 构建DataFrame =====================
# 横轴标签：每个攻击下面显示 5,10,15,20,25,30
x_labels = []
for attack in ATTACK_LABELS:
    for nm in NM_VALUES:
        x_labels.append(str(nm))

# 构建数据矩阵
data_matrix = []
for defense in DEFENSE_LIST:
    row = []
    for attack in ATTACK_LABELS:
        for nm in NM_VALUES:
            gain = recovery_data[defense][attack].get(nm)
            if gain is not None:
                row.append(gain)
            else:
                row.append(np.nan)
    data_matrix.append(row)

df_heatmap = pd.DataFrame(data_matrix, index=DEFENSE_LABELS, columns=x_labels)

print("\n" + "=" * 60)
print("Recovery Gain 数据矩阵:")
print(df_heatmap.round(2))
print("=" * 60)

# ===================== 5. 绘制热力图 =====================
print("\n绘制热力图...")

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5))

max_val = max(abs(df_heatmap.min().min()), abs(df_heatmap.max().max()))
vmax = max(max_val, 5)

cmap = sns.diverging_palette(250, 10, as_cmap=True, s=90, l=45)

sns.heatmap(
    df_heatmap,
    annot=True,
    fmt='.1f',
    cmap=cmap,
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    cbar_kws={
        'label': 'Recovery Gain (Accuracy%)',
        'shrink': 0.8,
        'aspect': 30
    },
    annot_kws={'fontsize': 8}
)

ax.set_xlabel('Number of Malicious Nodes', fontsize=11, fontweight='bold')
ax.set_ylabel('Defense Method', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)

# 分组分割线：每6列一组
split_positions = [6, 12, 18]
for pos in split_positions:
    ax.axvline(x=pos, color='black', linewidth=2, alpha=0.5)

# 在顶部标注攻击类型
attack_positions = [3, 9, 15, 21]
for i, attack in enumerate(ATTACK_LABELS):
    ax.text(attack_positions[i], -0.6, attack,
            ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax.transData, color='black')

plt.tight_layout()

plt.savefig('figures/recovery_gain_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/recovery_gain_heatmap.pdf', dpi=300, bbox_inches='tight')

print("\n" + "=" * 60)
print("✅ 热力图生成成功！")
print("   - PNG: figures/recovery_gain_heatmap.png")
print("   - PDF: figures/recovery_gain_heatmap.pdf")
print("=" * 60)

print("\n关键发现:")
print("-" * 40)
for defense in DEFENSE_LIST:
    gains = [g for g in recovery_data[defense].values() for g in g.values() if g is not None]
    if gains:
        avg_gain = np.mean(gains)
        max_gain = max(gains)
        min_gain = min(gains)
        neg_count = sum(1 for g in gains if g < 0)
        print(f"{defense:15s} | 平均: {avg_gain:6.2f}% | 最大: {max_gain:6.2f}% | 最小: {min_gain:6.2f}% | 负值: {neg_count:2d}/{len(gains)}")