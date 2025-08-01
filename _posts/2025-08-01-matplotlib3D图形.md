---
layout: post
title: "matplotlib-3D图形"
date: 2025-08-01
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- matplotlib
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>




Matplotlib 也可以绘制基本的三维图表，这对于可视化三维数据或函数非常有用。要绘制 3D 图形，需要导入 `mpl_toolkits.mplot3d`。

要创建 3D 图形，首先需要导入 `mpl_toolkits.mplot3d.axes3d` 模块中的 `Axes3D`。

创建 3D Axes 的两种常见方法：

1. **在 `Figure` 中添加 3D 子图**: `fig = plt.figure()` `ax = fig.add_subplot(111, projection='3d')` 这是推荐的面向对象方式。
2. **从现有 `Figure` 获取 3D Axes**: `fig = plt.figure()` `ax = Axes3D(fig)` 这种方式也可以创建 3D Axes，但 `fig.add_subplot(projection='3d')` 更常用，因为它直接将 Axes 添加到 Figure 的子图网格中。

# 1.三维折线图散点图 (3D Line and Scatter Plot)

在 3D Axes 上绘制折线图和散点图与 2D 类似，但需要提供 X, Y, Z 三个坐标。

- **三维折线图**: `ax.plot(xs, ys, zs, zdir='z', **kwargs)`
    - `xs`, `ys`, `zs`: 数据点的 X, Y, Z 坐标序列。
    - `zdir`: 绘制方向，默认为 `'z'`。
    - `**kwargs`: 与 2D 折线图类似的线条样式参数（如 `color`, `linewidth`, `linestyle`）。
- **三维散点图**: `ax.scatter(xs, ys, zs, zdir='z', s=20, c=None, depthshade=True, **kwargs)`
    - `xs`, `ys`, `zs`: 数据点的 X, Y, Z 坐标序列。
    - `zdir`: 绘制方向，默认为 `'z'`。
    - `s`: 标记大小。
    - `c`: 标记颜色。
    - `depthshade`: 布尔值，如果为 `True` (默认)，则根据深度对标记进行着色，使其看起来有景深效果。
    - `**kwargs`: 与 2D 散点图类似的参数（如 `marker`, `cmap`, `alpha`, `edgecolors`）。

**设置 3D 轴标签**: 使用 `ax.set_xlabel()`, `ax.set_ylabel()`, `ax.set_zlabel()` 来设置三维轴的标签。

---

```python
# 导包
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 导入3D引擎

# 1.准备数据
# 三维折线图数据（螺旋线）
t_line = np.linspace(- 4 * np.pi, 4*np.pi+0.00001, 200)
x_line = np.sin(t_line)
y_line = np.cos(t_line)
z_line = t_line

# 三维散点图数据
np.random.seed(42)
num_scatter_points = 100
x_scatter = np.random.rand(num_scatter_points) * 10 - 5
y_scatter = np.random.rand(num_scatter_points) * 10 - 5
z_scatter = np.random.rand(num_scatter_points) * 10 - 5
sizes_scatter = np.random.randint(20, 200, size=num_scatter_points)
colors_scatter = np.random.rand(num_scatter_points)

fig = plt.figure(figsize=(10, 8))

# 2.创建3D Axes
ax3d = fig.add_subplot(111, projection="3d")

# 3.绘制三维折线图
ax3d.plot(x_line, y_line, z_line, color="blue", lw=2, label="3D折线图")

# 4.绘制三维散点图
scatter_3d = ax3d.scatter(
    x_scatter, y_scatter, z_scatter, 
    s = sizes_scatter, 
    c = colors_scatter, 
    cmap="viridis", 
    alpha=0.7, 
    marker="o", 
    edgecolor="k", 
    lw=0.5,
    depthshade=True
)

# 5.设置3D轴标签
ax3d.set_xlabel("X 轴", fontsize=12, color="red")
ax3d.set_ylabel("Y 轴", fontsize=12, color="green")
ax3d.set_zlabel("Z 轴", fontsize=12, color="blue")

# 6.添加标题和图例
ax3d.set_title("3D折线图和散点图", fontsize=16)
ax3d.legend()

# 7.添加颜色条
cbar = fig.colorbar(scatter_3d, ax=ax3d, pad=0.1)
cbar.set_label("颜色值", rotation=270, labelpad=15)

# 8.设置初始视角（可选）
ax3d.view_init(elev=20, azim=45)# 仰角和方位角
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250729151202926.png" alt="image-20250729151202926" style="zoom:50%;" />

**选择题**

1. 要在 Matplotlib 中绘制 3D 图形，首先需要导入哪个模块？

   A. `matplotlib.pyplot` B. `numpy` C. `mpl_toolkits.mplot3d.axes3d` D. `scipy.stats`

   > 答案：C

2. 在 `ax.scatter()` 绘制三维散点图时，哪个参数用于根据深度对标记进行着色，使其看起来有景深效果？

   A. `s` B. `c` C. `depthshade` D. `marker`

   > 答案：C，`depthshade=True` (默认) 会根据 Z 轴深度调整标记的亮度，模拟景深。

**编程题**

1. 绘制一个三维折线图，表示一个螺旋线。
    - X 坐标: sin(t)
    - Y 坐标: cos(t)
    - Z 坐标: t
    - 其中 t 从 −2pi 到 2pi，包含 100 个点。
    - 曲线颜色为紫色，线条宽度为 2。
    - 添加标题 "3D Helix Plot"。
    - 添加 X, Y, Z 轴标签。

```python
# 导包
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 导入3D引擎

# 1.准备数据
# 三维折线图数据（螺旋线）
t_line = np.linspace(- 2 * np.pi, 2*np.pi+0.00001, 100)
x_line = np.sin(t_line)
y_line = np.cos(t_line)
z_line = t_line

fig = plt.figure(figsize=(10, 8))

# 2.创建3D Axes
ax3d = fig.add_subplot(111, projection="3d")

# 3.绘制三维折线图
ax3d.plot(x_line, y_line, z_line, color="purple", lw=2, label="螺旋线")

# 5.设置3D轴标签
ax3d.set_xlabel("X 轴", fontsize=12, color="red")
ax3d.set_ylabel("Y 轴", fontsize=12, color="green")
ax3d.set_zlabel("Z 轴", fontsize=12, color="blue")

# 6.添加标题和图例
ax3d.set_title("3D Helix Plot", fontsize=16)
ax3d.legend()



# 8.设置初始视角（可选）
ax3d.view_init(elev=20, azim=45)# 仰角和方位角
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250729151549595.png" alt="image-20250729151549595" style="zoom:50%;" />

# 2.三维柱状图 (3D Bar Chart)



三维柱状图用于在三维空间中表示离散数据，通常用于展示在二维平面上有多个类别，并且每个类别在第三个维度上有一个数值的情况。

主要函数是 `ax.bar3d(x, y, z, dx, dy, dz, color=None, zsort='average', **kwargs)`。

- `x`, `y`, `z`: 每个柱子的基座 (底面) 的 X, Y, Z 坐标。`z` 通常为 0，表示柱子从 XY 平面开始。
- `dx`, `dy`, `dz`: 每个柱子在 X, Y, Z 方向上的宽度、深度和高度。
- `color`: 柱子的颜色。
- `zsort`: 决定绘制顺序，以正确处理重叠。`'average'` (默认) 根据柱子中心点的平均 Z 坐标排序。
- `**kwargs`: 其他可选参数，如 `alpha`, `edgecolor` 等。

三维柱状图的难点在于正确设置 `x`, `y`, `z`, `dx`, `dy`, `dz` 参数来表示每个柱子的位置和尺寸。

```python
from mpl_toolkits.mplot3d.axes3d import Axes3D # 导入 3D 引擎

# 1. 准备数据
# 模拟每个月 4 周的销量数据
months = np.arange(1, 5) # 4个月
weeks = np.arange(4) # 每月 4 周

# 创建 X, Y 坐标网格 (每个柱子的基座位置)
# np.meshgrid 创建两个二维数组，用于生成所有 (x, y) 组合
X, Y = np.meshgrid(months, weeks)
# 将网格展平为一维数组，作为 bar3d 的 x, y 参数
x_flat = X.flatten()
y_flat = Y.flatten()

# 模拟每个柱子的高度 (销量)
# 这里生成 (4个月 * 4周) = 16 个随机销量
z_heights = np.random.randint(5, 25, size=len(x_flat))
z_base = np.zeros_like(z_heights) # 柱子从 Z=0 开始

# 柱子的宽度和深度
dx = 0.8
dy = 0.8

# 随机生成颜色
colors = plt.cm.viridis(z_heights / z_heights.max())

fig = plt.figure(figsize=(8, 6), dpi=150)
# 2. 创建 3D Axes
ax3d = fig.add_subplot(111, projection='3d')

# 3. 绘制三维柱状图
ax3d.bar3d(
    x_flat, # X 坐标
    y_flat, # Y 坐标
    z_base, # Z 坐标 (柱子底部)
    dx, # X 方向宽度
    dy, # Y 方向深度
    z_heights, # Z 方向高度
    color=colors, # 颜色
    alpha=0.8, # 透明度
    edgecolor='black' # 边缘颜色
)

# 4. 设置轴标签和标题
ax3d.set_xlabel('Month', fontsize=12, color='red')
ax3d.set_ylabel('Week', fontsize=12, color='green')
ax3d.set_zlabel('Sales', fontsize=12, color='blue')
ax3d.set_title('Monthly Sales by Week (3D Bar Chart)', fontsize=16)

# 5. 设置 X, Y 轴刻度标签
ax3d.set_xticks(months)
ax3d.set_yticks(weeks)

# 6. 设置初始视角 (可选)
ax3d.view_init(elev=25, azim=-45)

plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250729151905600.png" alt="image-20250729151905600" style="zoom:50%;" />

**选择题**

1. 在 `ax.bar3d(x, y, z, dx, dy, dz, ...)` 中，`dx`, `dy`, `dz` 分别表示什么？

   A. 柱子的起点坐标。 B. 柱子的终点坐标。 C. 柱子在 X, Y, Z 方向上的宽度、深度和高度。 D. 柱子的颜色。

   > 答案：C

2. `ax.bar3d()` 的 `zsort` 参数主要用于解决什么问题？

   A. 柱子的颜色选择。 B. 柱子的透明度设置。 C. 柱子的绘制顺序，以正确处理重叠和透视效果。 D. 柱子的标签显示。

   > 答案：C，在 3D 绘图中，`zsort` 确保了离观察者较远的物体被正确地绘制在较近的物体后面。

**编程题**

1. 模拟一个小型零售店在不同月份（"Jan", "Feb", "Mar"）和不同产品类别（"Electronics", "Apparel"）的销售额。
    - 销售额数据（假设为随机整数 100-500）：
        - Jan: Electronics (随机), Apparel (随机)
        - Feb: Electronics (随机), Apparel (随机)
        - Mar: Electronics (随机), Apparel (随机)
2. 绘制这些销售额的三维柱状图。
    - X 轴表示月份，Y 轴表示产品类别，Z 轴表示销售额。
    - 柱子宽度和深度均为 0.5。
    - 为柱子选择一个合适的颜色映射。
    - 添加标题 "Monthly Sales by Product Category"。
    - 添加 X, Y, Z 轴标签。

```python
from mpl_toolkits.mplot3d.axes3d import Axes3D # 导入 3D 引擎

# 1. 准备数据
months = ["Jan", "Feb", "Mar"]
type_ = ["Electronics", "Apparel"]

# 创建 X, Y 坐标网格 (每个柱子的基座位置)
# np.meshgrid 创建两个二维数组，用于生成所有 (x, y) 组合
x_pos = np.arange(len(months))
y_pos = np.arange(len(type_))
X, Y = np.meshgrid(x_pos, y_pos)
# 将网格展平为一维数组，作为 bar3d 的 x, y 参数

x_flat = X.flatten()
y_flat = Y.flatten()

# 模拟每个柱子的高度 (销量)
# 这里生成 (4个月 * 4周) = 16 个随机销量
z_heights = np.random.randint(100, 501, size=len(x_flat))
z_base = np.zeros_like(z_heights) # 柱子从 Z=0 开始

# 柱子的宽度和深度
dx = 0.5
dy = 0.5

# 随机生成颜色
colors = plt.cm.viridis(z_heights / z_heights.max())

fig = plt.figure(figsize=(8, 6), dpi=150)
# 2. 创建 3D Axes
ax3d = fig.add_subplot(111, projection='3d')

# 3. 绘制三维柱状图
ax3d.bar3d(
    x_flat, # X 坐标
    y_flat, # Y 坐标
    z_base, # Z 坐标 (柱子底部)
    dx, # X 方向宽度
    dy, # Y 方向深度
    z_heights, # Z 方向高度
    color=colors, # 颜色
    alpha=0.8, # 透明度
    edgecolor='black' # 边缘颜色
)

# 4. 设置轴标签和标题
ax3d.set_xlabel('Month', fontsize=12, color='red')
ax3d.set_ylabel('产品类型', fontsize=12, color='green')
ax3d.set_zlabel('Sales', fontsize=12, color='blue')
ax3d.set_title('Monthly Sales by Product Category', fontsize=16)

# 5. 设置 X, Y 轴刻度标签为实际名称
ax3d.set_xticks(x_pos + dx/2) # 刻度位置调整到柱子中心
ax3d.set_xticklabels(months)
ax3d.set_yticks(y_pos + dy/2) # 刻度位置调整到柱子中心
ax3d.set_yticklabels(type_)

# 6. 设置初始视角 (可选)
ax3d.view_init(elev=25, azim=-45)

plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250801134945346.png" alt="image-20250801134945346" style="zoom:50%;" />









