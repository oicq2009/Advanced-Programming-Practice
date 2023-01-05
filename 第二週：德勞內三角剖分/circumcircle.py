# 实现外接圆生成函数

import math
triangle = [] # 剖分三角形
center = []   # 外接圆中点
radius = []   # 外接圆半径
 
## 接收3点计算外接圆的函数
def circumcircle(p1, p2, p3): #, display = True):  
  ## 已知散点，计算外接圆坐标与半径。
  x = p1[0] + p1[1] * 1j
  y = p2[0] + p2[1] * 1j
  z = p3[0] + p3[1] * 1j
 
  w = (z - x) / (y - x)
  res = (x - y) * (w - abs(w)**2) / 2j / w.imag - x
 
  X = -res.real
  Y = -res.imag
  rad = abs(res + x)
 
  return X, Y, rad
 
c_x, c_y, radius = circumcircle(points[0], points[1], points[3])
print(c_x,c_y,radius)
 
 
## 显示结果
plt.figure(figsize=(4,4))
plt.scatter (
    points[:, 0], points[:, 1], 
    marker='o', s=100, edgecolor="k", linewidth=2)
 
M = 1000
angle = np.exp(1j * 2 * np.pi / M)
angles = np.cumprod(np.ones(M + 1) * angle)
x, y = c_x + radius * np.real(angles), c_y + radius * np.imag(angles)
plt.plot(x, y, c='b')
plt.scatter( [c_x], [c_y], s=25, c= 'b')
plt.xlim([0, 700])
plt.ylim([0, 700])
plt.show()
