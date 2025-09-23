import csv

# 输入输出文件路径
input_file = "workload_decline.csv"
output_file = "output.csv"

# 每小时任务分布
hourly_counts = [10, 12, 14, 15, 10, 12, 12, 14, 16, 22, 30, 40, 50, 60, 45, 50, 55, 60, 65, 70, 55, 30, 20, 13]

# 生成144个时间桶的任务数
time_buckets = []
for count in hourly_counts:
    base = count // 6
    rem = count % 6
    time_buckets.extend([base + 1] * rem + [base] * (6 - rem))

# 读取原始数据并处理列名
rows = []
with open(input_file, 'r') as f:
    # 读取第一行获取列名
    header_line = f.readline().strip()
    fieldnames = header_line.split(',')

    # 重置文件指针并创建DictReader
    f.seek(0)
    reader = csv.DictReader(f, delimiter=',')

    # 验证submit_time字段是否存在
    if 'submit_time' not in fieldnames:
        print(fieldnames)
        raise ValueError("CSV文件缺少'submit_time'列")

    # 读取所有行
    for row in reader:
        rows.append(row)

# 修改submit_time
task_idx = 0
for time_bucket, count in enumerate(time_buckets):
    for _ in range(count):
        if task_idx < len(rows):
            rows[task_idx]['submit_time'] = str(time_bucket)  # 转换为字符串
            task_idx += 1

# 验证总任务数
total_tasks = sum(time_buckets)
if task_idx != len(rows):
    print(f"警告: 时间桶总任务数({total_tasks})与CSV行数({len(rows)})不匹配")
    print(f"已处理{task_idx}个任务, 剩余{len(rows) - task_idx}个任务未分配时间")

# 写入新文件
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
    writer.writeheader()
    writer.writerows(rows)

print(f"处理完成! 共修改了{task_idx}个任务的submit_time")