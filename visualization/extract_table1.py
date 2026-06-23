import os
import re
import pandas as pd
from pathlib import Path

LOG_DIR = "../supplemental_experiments"

# 定义映射 - 更新防御名称映射
ATTACK_MAP = {
    "gaussian": "GA",
    "zero_gradient": "ZG",
    "sign_flipping": "SF",
    "shifted_mean": "MS",
}

DEFENSE_MAP = {
    "mandera_detect": "MANDERA",
    "multi_krum": "Krum",
    "bulyan": "Bulyan",
    "median": "Median",
    "tr_mean": "Trim-mean",
    "fltrust": "FLTrust",
}

def parse_filename(filename):
    """解析文件名: {attack}_attack_{defense}_{nm}.log"""
    name = filename.replace('.log', '')
    
    # 模式: gaussian_attack_mandera_detect_15
    pattern = r'^([a-zA-Z_]+)_attack_([a-zA-Z_]+)_(\d+)$'
    match = re.match(pattern, name)
    if match:
        attack_key = match.group(1).lower()
        defense_key = match.group(2).lower()
        nm = int(match.group(3))
        return attack_key, defense_key, nm
    
    return None

def parse_log_file(filepath):
    """解析日志文件，返回最后一轮的准确率"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有准确率行
        pattern = r'Test set: Accuracy: (\d+)/(\d+) \((\d+)%\)'
        matches = re.findall(pattern, content)
        
        if not matches:
            return None
        
        correct, total, acc = matches[-1]
        return int(correct) / int(total)
    except Exception as e:
        return None

def main():
    # 检查目录
    if not os.path.exists(LOG_DIR):
        print(f"警告: 目录 {LOG_DIR} 不存在")
        print(f"当前工作目录: {os.getcwd()}")
        return

    # 收集所有日志文件
    all_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    print(f"在 {LOG_DIR} 中找到 {len(all_files)} 个日志文件")
    
    # 解析所有文件
    parsed_rows = []
    for filename in all_files:
        filepath = os.path.join(LOG_DIR, filename)
        parsed = parse_filename(filename)
        if parsed is None:
            continue
        
        attack_key, defense_key, nm = parsed
        
        # 映射到标准名称
        attack = ATTACK_MAP.get(attack_key)
        defense = DEFENSE_MAP.get(defense_key)
        
        if attack is None or defense is None:
            print(f"跳过无法映射的文件: {filename} (attack={attack_key}, defense={defense_key})")
            continue
        
        # 解析准确率
        acc = parse_log_file(filepath)
        if acc is not None:
            parsed_rows.append({
                'filename': filename,
                'attack': attack,
                'defense': defense,
                'nm': nm,
                'accuracy': acc,
                'accuracy_pct': acc * 100
            })
            print(f"✓ {attack} | {defense} | nm={nm} | {acc*100:.2f}%")
        else:
            print(f"✗ 解析失败: {filename}")
    
    print(f"\n成功解析 {len(parsed_rows)} 个文件")
    
    if len(parsed_rows) == 0:
        print("没有成功解析任何文件")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(parsed_rows)
    
    # 保存所有数据
    df.to_csv("table1_all_data.csv", index=False)
    print(f"所有数据已保存到: table1_all_data.csv")
    
    # 生成表格
    attacks = ['GA', 'ZG', 'SF', 'MS']
    defenses = ['MANDERA', 'Krum', 'Bulyan', 'Median', 'Trim-mean', 'FLTrust']
    nms = [5, 10, 15, 20, 25, 30]
    
    print("\n" + "="*100)
    print("Table 1: Final Accuracy (%) after 25 epochs")
    print("="*100)
    
    # 创建完整的表格数据
    table_data = {}
    for attack in attacks:
        table_data[attack] = {}
        for defense in defenses:
            table_data[attack][defense] = {}
            for nm in nms:
                matches = df[(df['attack'] == attack) & 
                            (df['defense'] == defense) & 
                            (df['nm'] == nm)]
                if not matches.empty:
                    table_data[attack][defense][nm] = matches['accuracy_pct'].mean()
                else:
                    table_data[attack][defense][nm] = None
    
    # 打印表格
    for attack in attacks:
        print(f"\nAttack: {attack}")
        print("-"*90)
        print(f"{'Defense':<12}", end='')
        for nm in nms:
            print(f"{nm:>8}", end='')
        print()
        print("-"*90)
        
        for defense in defenses:
            print(f"{defense:<12}", end='')
            has_data = False
            for nm in nms:
                val = table_data[attack][defense][nm]
                if val is not None:
                    print(f"{val:8.2f}", end='')
                    has_data = True
                else:
                    print(f"{'N/A':>8}", end='')
            print()
            if not has_data:
                print("  (没有数据)")
    
    # 保存为CSV格式的表格
    print("\n保存表格数据...")
    table_rows = []
    for attack in attacks:
        for defense in defenses:
            row = {'Attack': attack, 'Defense': defense}
            for nm in nms:
                val = table_data[attack][defense][nm]
                row[f'nm_{nm}'] = f"{val:.2f}" if val is not None else "N/A"
            table_rows.append(row)
    
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv("table1_formatted.csv", index=False)
    print("表格已保存到: table1_formatted.csv")
    
    # 保存为Markdown格式
    with open("table1_results.md", 'w') as f:
        f.write("# Table 1: Final Accuracy (%) after 25 epochs\n\n")
        f.write("| Attack | Defense | " + " | ".join([f"nm={nm}" for nm in nms]) + " |\n")
        f.write("|" + "|".join(["---------" for _ in range(len(nms) + 2)]) + "|\n")
        for attack in attacks:
            for defense in defenses:
                row = [attack, defense]
                for nm in nms:
                    val = table_data[attack][defense][nm]
                    row.append(f"{val:.2f}" if val is not None else "N/A")
                f.write("| " + " | ".join(row) + " |\n")
    
    print("Markdown表格已保存到: table1_results.md")
    
    # 打印统计信息
    print("\n" + "="*100)
    print("数据统计:")
    print(f"总文件数: {len(all_files)}")
    print(f"成功解析: {len(parsed_rows)}")
    print("="*100)

if __name__ == "__main__":
    main()
