'''
我们先学习如何生成散点，我们一般用 matplotlib 库中的 scatter 函数生成散点图。
代码演示：绘制一个带有四个散点的散点图
'''

# from sklearn.datasets import make_classification
import numpy as np
import matplotlib .pyplot as plt
 
points = [
    (120,240), (370,180), (550,460), (260,540)  # 定义散点的坐标
]
points = np.array(points) # 输出到图像
 
plt.title("data")         # 设置图像标题
 
## 利用 Matplotlib 中的 scatter 函数绘制散点图
plt.scatter (
    points[:, 0],         # 对应散点图中每个点的 x 坐标
    points[:, 1],         # 对于散点图中每个点的 y 坐标
    marker = 'o',         # maker参数指定散点图中每个散点的形状
    s = 100,              # 指定了散点图中每个散点的大小
    edgecolor = "k",      # 指定了散点图中每个散点的边缘颜色。
    linewidth = 2         # 参数指定了散点图中每个散点边缘的线宽。
)
 
plt.xlabel("$X$")         # 设置x坐标轴标签
plt.ylabel("$Y$")         # 设置y坐标轴标签
plt.show()                # 显示图像
