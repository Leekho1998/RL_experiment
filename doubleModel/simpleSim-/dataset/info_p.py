import re
import csv
from collections import defaultdict


def count_tasks_per_group(file_path, output_csv):
    # 初始化字典用于存储每组任务数量
    group_counts = defaultdict(int)

    # 正则表达式匹配起始时间
    pattern = re.compile(r':\s*(\d+)\s*->')

    with open(file_path, 'r') as file:
        for line in file:
            # 提取起始时间
            match = pattern.search(line)
            if match:
                start_time = int(match.group(1))
                # 计算所属组号（每6个时间步一组）
                group_id = start_time // 6
                # 增加该组的计数
                group_counts[group_id] += 1

    # 准备CSV数据
    csv_data = []
    for group_id in sorted(group_counts.keys()):
        start_range = group_id * 6
        end_range = start_range + 5
        csv_data.append({
            "Group ID": group_id,
            "Time Range": f"{start_range}-{end_range}",
            "Task Count": group_counts[group_id]
        })

    # 写入CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["Group ID", "Time Range", "Task Count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)

    print(f"统计结果已保存到: {output_csv}")


# 使用示例
count_tasks_per_group('info.txt', 'task_group_counts.csv')