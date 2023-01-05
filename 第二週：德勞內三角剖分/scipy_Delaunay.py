# 利用 scipy 库的 Delaunay 函数

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
 
## 随机生成点集
points = np.random.rand(30, 2)
 
## 计算德劳内，使用scipy库中的 Delaunay函数计算
tri = Delaunay(points)
 
# 画图
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
