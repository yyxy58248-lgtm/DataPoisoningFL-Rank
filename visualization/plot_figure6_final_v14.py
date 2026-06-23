import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

# ===================== 全局绘图样式 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# ===================== 核心配置 =====================
LOG_DIR = "../supplemental_experiments"

DEFENSE_KEYS = ['multi_krum', 'bulyan', 'tr_mean', 'fltrust', 'none', 'median', 'mandera_detect']
DEFENSE_LABELS = ['Krum', 'Bulyan', 'Trim-mean', 'FLTrust', 'NO-attack', 'Median', 'MANDERA']
DEFENSE_COLORS = ['#999999', '#66b3ff', '#ffd966', '#cccccc', '#003399', '#99d6a6', '#ff0000']
DEFENSE_LINESTYLES = ['-', '--', '-', ':', '-', '-.', '-']
DEFENSE_LINEWIDTHS = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8]

ATTACKS = ['GA', 'ZG', 'SF', 'MS']
ATTACK_NAMES = {
    'GA': 'gaussian',
    'ZG': 'zero_gradient',
    'SF': 'sign_flipping',
    'MS': 'shifted_mean'
}
ATTACK_LABELS = ['GA', 'ZG', 'SF', 'MS']
NM_VALUES = [5, 10, 15, 20, 25, 30]

ACC_YLIM = {'GA': (0, 80), 'ZG': (0, 80), 'SF': (62, 78), 'MS': (62, 78)}
ACC_YTICKS = {'GA': [0, 20, 40, 60, 80], 'ZG': [0, 20, 40, 60, 80],
              'SF': [40, 50, 60, 70, 80], 'MS': [40, 50, 60, 70, 80]}
LOSS_YLIM = {'GA': (0, 10.5), 'ZG': (0.4, 2.5), 'SF': (0.4, 2.5), 'MS': (0.4, 2.5)}
LOSS_YTICKS = {'GA': [0, 2.5, 5.0, 7.5, 10.0], 'ZG': [0.5, 1.0, 1.5, 2.0],
               'SF': [0.5, 1.0, 1.5, 2.0], 'MS': [0.5, 1.0, 1.5, 2.0]}

MAX_EPOCH = 25

def parse_accuracy_loss_from_log(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        acc_matches = re.findall(r'Test set: Accuracy: \d+/\d+ \((\d+)%\)', content)
        loss_matches = re.findall(r'Test set: Loss: ([\d\.]+)', content)
        acc = [float(m) for m in acc_matches] if acc_matches else []
        loss = [float(m) for m in loss_matches] if loss_matches else []
        acc = acc[:MAX_EPOCH]
        loss = loss[:MAX_EPOCH]
        return acc, loss
    except:
        return [], []

def find_log_file(attack_name, defense, nm):
    if defense == 'none':
        fp = os.path.join(LOG_DIR, f"no_attack_0.log")
        return fp if os.path.exists(fp) else None
    fp = os.path.join(LOG_DIR, f"{attack_name}_attack_{defense}_{nm}.log")
    return fp if os.path.exists(fp) else None

print("Generating Figure 6...")
all_data = {}
for attack in ATTACKS:
    all_data[attack] = {}
    for nm in NM_VALUES:
        all_data[attack][nm] = {}
        for defense in DEFENSE_KEYS:
            log_file = find_log_file(ATTACK_NAMES[attack], defense, nm)
            if log_file:
                acc, loss = parse_accuracy_loss_from_log(log_file)
                all_data[attack][nm][defense] = {'acc': acc, 'loss': loss}
            else:
                all_data[attack][nm][defense] = {'acc': [], 'loss': []}

epochs = np.arange(1, MAX_EPOCH + 1)

# ===================== 核心绘图 =====================
# 减小画布宽度，让子图更紧凑
fig = plt.figure(figsize=(10, 14))

gs_outer = fig.add_gridspec(2, 1, hspace=0.15, height_ratios=[1, 1])

# ---------------- 第一块：Accuracy 4行6列 ----------------
gs_acc = gs_outer[0, 0].subgridspec(4, 6, wspace=0.03, hspace=0.04)
for row, attack in enumerate(ATTACKS):
    for col, nm in enumerate(NM_VALUES):
        ax = fig.add_subplot(gs_acc[row, col])
        for d_idx, defense in enumerate(DEFENSE_KEYS):
            data = all_data[attack][nm].get(defense, {})
            acc = data.get('acc', [])
            if len(acc) > 0:
                ax.plot(epochs[:len(acc)], acc,
                       color=DEFENSE_COLORS[d_idx],
                       linestyle=DEFENSE_LINESTYLES[d_idx],
                       linewidth=DEFENSE_LINEWIDTHS[d_idx],
                       alpha=0.85, solid_capstyle='round')
        ax.set_xlim(0, MAX_EPOCH)
        ax.set_ylim(ACC_YLIM[attack][0], ACC_YLIM[attack][1])
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_yticks(ACC_YTICKS[attack])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        ax.set_facecolor('white')
        # 强制正方形，子图容器为正方形
        ax.set_box_aspect(1)

        if row == 0:
            ax.set_title(f'{nm}', fontsize=9, backgroundcolor='#f0f0f0', pad=2)
        if col == 5:
            ax.text(1.04, 0.5, ATTACK_LABELS[row], transform=ax.transAxes,
                   fontsize=10, va='center', ha='left', color='gray', rotation=90)
        if row != 3:
            ax.set_xticklabels([])
        if col != 0:
            ax.set_yticklabels([])

fig.text(0.015, 0.72, 'Accuracy', ha='center', va='center', 
         fontsize=12, fontweight='bold', rotation=90)

# ---------------- 第二块：log(Loss) 4行6列 ----------------
gs_loss = gs_outer[1, 0].subgridspec(4, 6, wspace=0.03, hspace=0.04)
for row, attack in enumerate(ATTACKS):
    for col, nm in enumerate(NM_VALUES):
        ax = fig.add_subplot(gs_loss[row, col])
        for d_idx, defense in enumerate(DEFENSE_KEYS):
            data = all_data[attack][nm].get(defense, {})
            loss = data.get('loss', [])
            if len(loss) > 0:
                log_loss = [np.log(l) if l > 0 else 0 for l in loss]
                ax.plot(epochs[:len(log_loss)], log_loss,
                       color=DEFENSE_COLORS[d_idx],
                       linestyle=DEFENSE_LINESTYLES[d_idx],
                       linewidth=DEFENSE_LINEWIDTHS[d_idx],
                       alpha=0.85, solid_capstyle='round')
        ax.set_xlim(0, MAX_EPOCH)
        ax.set_ylim(LOSS_YLIM[attack][0], LOSS_YLIM[attack][1])
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_yticks(LOSS_YTICKS[attack])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        ax.set_facecolor('white')
        ax.set_box_aspect(1)

        if row == 0:
            ax.set_title(f'{nm}', fontsize=9, backgroundcolor='#f0f0f0', pad=2)
        if col == 5:
            ax.text(1.04, 0.5, ATTACK_LABELS[row], transform=ax.transAxes,
                   fontsize=10, va='center', ha='left', color='gray', rotation=90)
        if row != 3:
            ax.set_xticklabels([])
        if col != 0:
            ax.set_yticklabels([])

fig.text(0.015, 0.28, 'log(Loss)', ha='center', va='center', 
         fontsize=12, fontweight='bold', rotation=90)

# ===================== X轴标签 =====================
fig.text(0.5, 0.50, 'Number of Epoch', ha='center', va='center', fontsize=10)
fig.text(0.5, 0.04, 'Number of Epoch', ha='center', va='center', fontsize=10)

# ===================== 图例 =====================
legend_elements = [plt.Line2D([0], [0], color='none', label='Defence')]
for d_idx, defense in enumerate(DEFENSE_KEYS):
    legend_elements.append(
        plt.Line2D([0], [0],
                   color=DEFENSE_COLORS[d_idx],
                   linestyle=DEFENSE_LINESTYLES[d_idx],
                   linewidth=DEFENSE_LINEWIDTHS[d_idx],
                   label=DEFENSE_LABELS[d_idx])
    )

fig.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 0.95), ncol=8, fontsize=7.5,
           frameon=False, handlelength=1.5)

fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 0.49), ncol=8, fontsize=7.5,
           frameon=False, handlelength=1.5)

plt.subplots_adjust(top=0.92, bottom=0.06, left=0.04, right=0.96)

plt.savefig('figure6_square.png', dpi=300)
plt.savefig('figure6_square.pdf', dpi=300)

print("="*60)
print("Figure 6 生成成功！")
print("解决列间距过大问题：")
print("1. 减小figsize宽度从18到10")
print("2. 保持set_box_aspect(1)保证正方形")
print("3. wspace=0.03精确控制列间距")
print("4. 子图自然紧凑排列")
print("="*60)
