import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

#1.读csv文件
df = pd.read_csv("iris.csv")
print("前5行数据:")
print(df.head())
print()

#2.查看基本信息
print("列名：",df.columns.tolist())
print("数据类型：")
print(df.dtypes)
print()

#3.基本统计量
print("基本统计量：")
print(df.describe())
print()

#4.按类别分组计算均值
print("按品种分组的花瓣长度均值：")
print(df.groupby("variety")["petal.length"].mean())
print()

#5.画图：散点图矩阵
#先看品种有哪些
print("品种：",df["variety"].unique())

#按品种着色画散点图（花萼长度 vs 花瓣长度）
plt.figure(figsize=(8,6))
for species in df["variety"].unique():
    subset = df[df["variety"] == species]
    plt.scatter(subset["sepal.length"], subset["petal.length"],label=species,alpha=0.7)

plt.xlabel("花萼长度 (sepal length)")
plt.ylabel("花瓣长度 (petal length)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("iris_scatter.png", dpi=150)
print("散点图已保存为 iris_scatter.png")
