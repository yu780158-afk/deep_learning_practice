import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 1. 生成 1000 个正态分布的随机数（均值=0，标准差=1）
data = np.random.randn(1000)

# 2. 计算均值和标准差
mean = np.mean(data)
std = np.std(data)

print(f"均值：{mean:.3f}")
print(f"标准差：{std:.3f}")

# 3. 画直方图
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title(f"直方图 (均值={mean:.3f}, 标准差={std:.3f})")
plt.xlabel("值")
plt.ylabel("频数")

# 4. 画折线图
plt.subplot(1,2,2)
plt.plot(data[:100], marker='o', linestyle='-', markersize=3)
plt.title=("前100个数据点折线图")
plt.xlabel("索引")
plt.ylabel("值")

# 5. 保存图片
plt.tight_layout()
plt.savefig("random_data_viz.png", dpi=150)
print("图片已保存为 random_data_viz.png")
