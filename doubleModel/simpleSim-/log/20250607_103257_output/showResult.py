# 首先，我们需要读取用户上传的.log文件来查找每个episode的"Total cost"。
# 打开并读取文件内容
import matplotlib.pyplot as plt

file_path = './output.log'

# 读取文件
with open(file_path, 'r') as file:
    log_content = file.readlines()

# 提取每个episode的"Total cost"
total_costs = []
Total_simulation_steps = []
Average_task_duration = []
Average_waiting_time = []
Average_running_time = []
Total_energy_consumption = []
for line in log_content:
    if "Total cost" in line:
        # 分割行以找到成本值
        parts = line.split(':')
        if len(parts) > 1:
            # 尝试将成本值转换为浮点数并添加到列表中
            try:
                cost = float(parts[1].strip())
                total_costs.append(cost)
            except ValueError:
                pass  # 如果转换失败，忽略该行
    if "Total simulation steps"in line:
        # 分割行以找到成本值
        parts = line.split(':')
        if len(parts) > 1:
            # 尝试将成本值转换为浮点数并添加到列表中
            try:
                cost = float(parts[1].strip())
                Total_simulation_steps.append(cost)
            except ValueError:
                pass  # 如果转换失败，忽略该行
    if "Total energy consumption" in line:
        # 分割行以找到成本值
        parts = line.split(':')
        if len(parts) > 1:
            # 尝试将成本值转换为浮点数并添加到列表中
            try:
                cost = float(parts[1].strip())
                Total_energy_consumption.append(cost)
            except ValueError:
                pass  # 如果转换失败，忽略该行


# 检查是否成功提取了100个Total cost值
len(total_costs), total_costs[:5]  # 显示提取的总数和前5个成本值以进行验证


# 现在我们有了所有episode的Total cost，我们可以绘制曲线图
episode_numbers = range(1, 101)  # Episode编号从1到100

# 绘制曲线图
plt.figure(figsize=(10, 5))
plt.plot(episode_numbers, total_costs, marker='o', linestyle='-', color='b')
plt.title('Total Cost per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Cost')
plt.grid(True)
plt.tight_layout()

# 保存曲线图
curve_image_path = './total_cost_per_episode.png'
plt.savefig(curve_image_path)

# 显示曲线图路径
curve_image_path


plt.figure(figsize=(10, 5))
plt.plot(episode_numbers, Total_simulation_steps, marker='o', linestyle='-', color='g')

plt.title('Total steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Total steps')
plt.grid(True)
plt.tight_layout()

# 保存曲线图
curve_image_path = './total_steps_per_episode.png'
plt.savefig(curve_image_path)



plt.figure(figsize=(10, 5))
plt.plot(episode_numbers, Total_energy_consumption, marker='o', linestyle='-', color='r')
plt.title('Total energy Cost per Episode')
plt.xlabel('Episode')
plt.ylabel('Total energy Cost')
plt.grid(True)
plt.tight_layout()

# 保存曲线图
curve_image_path = './total_energy_per_episode.png'
plt.savefig(curve_image_path)