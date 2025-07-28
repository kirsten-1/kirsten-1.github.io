---
layout: post
title: "matplotlib-文本注释箭头"
date: 2025-07-28
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- matplotlib
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>




在 Matplotlib 中，添加文本、注释和箭头是使图表更具信息量和解释性的关键。它们可以帮助我们突出关键数据点、解释趋势或提供额外的上下文信息。

常用函数如下：

| **Pyplot函数** | **API方法**                    | **描述**                           |
| -------------- | ------------------------------ | ---------------------------------- |
| `text()`       | `mpl.axes.Axes.text()`         | 在 `Axes` 对象的任意位置添加文字   |
| `xlabel()`     | `mpl.axes.Axes.set_xlabel()`   | 为 X 轴添加标签                    |
| `ylabel()`     | `mpl.axes.Axes.set_ylabel()`   | 为 Y 轴添加标签                    |
| `title()`      | `mpl.axes.Axes.set_title()`    | 为 `Axes` 对象添加标题             |
| `legend()`     | `mpl.axes.Axes.legend()`       | 为 `Axes` 对象添加图例             |
| `annotate()`   | `mpl.axes.Axes.annotate()`     | 为 `Axes` 对象添加注释（箭头可选） |
| `figtext()`    | `mpl.figure.Figure.text()`     | 在 `Figure` 对象的任意位置添加文字 |
| `suptitle()`   | `mpl.figure.Figure.suptitle()` | 为 `Figure` 对象添加中心化的标题   |

# 1.文本

文本是图表中最基本的信息载体。Matplotlib 提供了多种函数来添加不同类型的文本：

1. **`plt.text(x, y, s, fontdict=None, **kwargs)` / `ax.text(x, y, s, fontdict=None, **kwargs)`**:
    - 在指定的 `(x, y)` 数据坐标位置添加文本。
    - `x`, `y`: 文本的起始 X 和 Y 坐标（在数据坐标系中）。
    - `s`: 要显示的字符串内容。
    - `fontdict`: 一个字典，用于设置字体属性，例如 `fontsize`, `family`, `color`, `weight` 等。
    - `**kwargs`: 其他文本属性，如 `horizontalalignment` (ha), `verticalalignment` (va), `rotation` 等。
    - `plt.text()` 作用于当前活动的 `Axes`，`ax.text()` 直接作用于指定的 `Axes`。
2. **`plt.title(s, fontdict=None, loc='center', **kwargs)` / `ax.set_title(s, fontdict=None, loc='center', **kwargs)`**:
    - 设置 `Axes` 的标题。
    - `s`: 标题字符串。
    - `loc`: 标题的位置，可以是 `'left'`, `'center'`, `'right'`。
    - `fontdict`: 字体属性字典。
3. **`plt.xlabel(s, fontdict=None, labelpad=4, **kwargs)` / `ax.set_xlabel(s, fontdict=None, labelpad=4, **kwargs)`**:
    - 设置 X 轴的标签。
    - `labelpad`: 标签与轴的距离。
4. **`plt.ylabel(s, fontdict=None, labelpad=4, **kwargs)` / `ax.set_ylabel(s, fontdict=None, labelpad=4, **kwargs)`**:
    - 设置 Y 轴的标签。
5. **`plt.suptitle(s, x=0.5, y=0.98, fontdict=None, **kwargs)` / `fig.suptitle(s, x=0.5, y=0.98, fontdict=None, **kwargs)`**:
    - 设置整个 `Figure` 的标题，通常位于 Figure 的顶部中心。
    - `x`, `y`: 标题在 Figure 坐标系中的位置（比例值）。
    - `fontdict`: 字体属性字典。
6. **`plt.figtext(x, y, s, fontdict=None, **kwargs)` / `fig.text(x, y, s, fontdict=None, **kwargs)`**:
    - 在 `Figure` 的任意位置添加文本。与 `plt.text()` 类似，但坐标 `(x, y)` 是 Figure 坐标系中的比例值（0到1）。
    - 常用于添加版权信息、数据来源或全局性的说明。

**字体属性 (`fontdict`)**： `fontdict` 是一个字典，可以包含以下常用键值对：

- `'fontsize'`: 字体大小 (例如 12, 'large', 'x-small')。
- `'family'`: 字体家族 (例如 'serif', 'sans-serif', 'monospace', 'Kaiti SC' 等)。
- `'color'`: 字体颜色。
- `'weight'`: 字体粗细 (例如 'normal', 'bold', 'light', 'heavy')。
- `'style'`: 字体样式 (例如 'normal', 'italic', 'oblique')。

**LaTeX 支持**： Matplotlib 对 LaTeX 语法有很好的支持，可以在文本字符串中使用 LaTeX 表达式来显示数学公式。只需将字符串用 `r'$...$'` 包裹起来即可，其中 `r` 表示原始字符串，避免反斜杠的转义问题。例如 `r'$\alpha + \beta^2$'`。

> Matplotlib 中的所有文本都是 `matplotlib.text.Text` 类的实例。当你在图表中添加文本时，Matplotlib 会创建一个 `Text` 对象，并将其作为 `Artist` 添加到 `Axes` 或 `Figure` 中。这些 `Text` 对象拥有自己的坐标系统（数据坐标、轴坐标或 Figure 坐标），并且可以独立地进行定位、旋转、着色和字体设置。
>
> 对于 LaTeX 文本，Matplotlib 会使用其内置的数学文本渲染器 (通常是 `usetex=False` 时的 `mathtext` 引擎，或 `usetex=True` 时的外部 LaTeX 编译器) 将 LaTeX 表达式转换为图形。

**文本对齐**:

- `ha` (horizontalalignment): `'left'`, `'center'`, `'right'`
- `va` (verticalalignment): `'top'`, `'center'`, `'bottom'`, `'baseline'`

**文本旋转**: `rotation` 参数可以设置文本的旋转角度。

**文本框**: 可以通过 `bbox` 参数为文本添加背景框，例如 `bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5)`。

**全局字体设置**: 可以通过 `plt.rcParams` 来全局设置字体，例如 `plt.rcParams['font.family'] = 'SimHei'` (用于中文显示，可能需要配置 Matplotlib 字体缓存)。

**图例 (`plt.legend()`)**: 图例也是一种特殊的文本，用于标识图中的不同数据系列。它会自动收集 `plot()` 函数中 `label` 参数定义的标签。

**应用场景**:

- **数据点标注**: 在散点图中，为特定的数据点添加文本标签，如异常值、关键事件点。
- **趋势线说明**: 在回归分析图中，标注回归方程或 R-squared 值。
- **图表元数据**: 在图表底部添加数据来源、作者信息、日期等。
- **机器学习模型结果解释**: 在决策边界图中，标注不同区域代表的类别；在特征重要性图中，直接在条形图上显示重要性数值。

---

```python
# 定义字体属性字典
font_title = {
    'fontsize': 24,
    "family": "SimHei", 
    "color": "darkblue"
}

font_text = {
    'fontsize': 16,
    "family": "SimHei", 
    "color": "green"
}
x = np.linspace(0, 0.5+0.00001, 100)
y = np.cos(2*np.pi * x)*np.exp(-x)

plt.figure(figsize=(10, 7))
ax = plt.gca()

ax.plot(x, y, "k-", lw=2, label="衰减余弦波")
ax.set_title("标题：衰减余弦波", fontdict=font_title, loc="center")

plt.suptitle("Figure标题", y = 1.02, fontdict=font_title, fontsize=28)

# 在axes中添加普通文本
plt.text(x = 0.2, y=0.65, 
        s = r"$cos(2 \pi x)*e^{-x}$", 
        fontdict=font_text, 
        ha = "center",   # 水平居中对齐
        va = "bottom",  # 垂直底部对齐
         bbox = dict(boxstyle="round", pad=3, fc="yellow", ec="red", lw=1, alpha=0.6))  # 文本框设置
# 添加X轴和Y轴标签
ax.set_xlabel("Time(s)", fontsize=14, color="gray", labelpad=10)
ax.set_ylabel("Voltage(mV)", fontsize=14, color="gray", labelpad=10)

# 添加图例
ax.legend(loc="upper right", fontsize=12)


# 在Figure中添加文本
plt.figtext(0.98, 0.02,
           "Data Source: Simulated", fontsize=10, color="darkgreen", ha="right", va="bottom",
           bbox=dict(boxstyle="square, pad=0.1", fc="lightgray", ec="none", alpha=0.7))

# 调整布局
plt.tight_layout(rect=[0,0.05,1,0.98])
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727220727925.png" alt="image-20250727220727925" style="zoom:50%;" />

**选择题**

1. 要在 Matplotlib 图表中显示数学公式，应将公式字符串用什么包裹起来？

   A. `'...'` B. `r'...'` C. `'$...$'` D. `r'$...$'`

   > 答案：D，`r''` 表示原始字符串，避免反斜杠转义。`$...$` 表示 LaTeX 数学模式。两者结合才能正确渲染数学公式。

2. `plt.suptitle()` 和 `ax.set_title()` 的主要区别是什么？

   A. `plt.suptitle()` 设置 Figure 标题，`ax.set_title()` 设置 Axes 标题。

   B. `plt.suptitle()` 只能设置字体大小，`ax.set_title()` 可以设置所有字体属性。

   C. `plt.suptitle()` 只能在图表中心，`ax.set_title()` 可以任意位置。

   D. 它们没有区别，是完全等效的。

   > 答案：A，`suptitle` 是 Figure 级别的标题，位于整个图表的顶部。`set_title` 是 Axes 级别的标题，位于单个子图的顶部。

**编程题**

1. 创建一个简单的折线图，绘制 $$y=x^2$$ 在 $$x \in [−5,5]$$ 范围内的曲线。
2. 为图表添加以下文本元素：
    - Figure 标题："Quadratic Function Analysis"，字体大小 20，颜色蓝色。
    - Axes 标题："Parabola Plot"，字体大小 16，颜色黑色。
    - X 轴标签："Input Value (x)"。
    - Y 轴标签："Output Value (y)"。
    - 在图中的点 `(0, 0)` 处添加文本 "Vertex"，字体大小 14，颜色红色，并用一个圆角矩形框住。
    - 在 Figure 的左下角添加文本 "Generated by Matplotlib"，字体大小 10，颜色灰色。

```python
x = np.linspace(-5, 5+0.0001, 100)
y = x**2
ax = plt.gca()
ax.plot(x, y, color="pink")

plt.suptitle("Quadratic Function Analysis", fontsize=20, color="blue")
ax.set_title("Parabola Plot", fontsize=16, color="k")
ax.set_xlabel("Input Value (x)")
ax.set_ylabel("Output Value (y)")

plt.text(x=0, y=0, s="Vertex", fontsize=14, color="red", 
         bbox=dict(boxstyle="round", alpha=0.5))
plt.figtext(x = 0, y = 0, s = "Generated by Matplotlib", fontsize=10, color="gray")

plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727222427587.png" alt="image-20250727222427587" style="zoom:50%;" />





# 2.箭头 (`plt.arrow`)

`plt.arrow(x, y, dx, dy, **kwargs)` 函数用于在图表中绘制简单的直线箭头。它直接指定了箭头的起点和箭头的长度及方向。

- `x, y`: 箭头的起点坐标（数据坐标）。
- `dx, dy`: 箭头的 X 和 Y 方向上的长度。箭头的终点是 `(x + dx, y + dy)`。
- `**kwargs`: 其他可选参数，用于控制箭头的样式：
    - `head_width`: 箭头头部的宽度。
    - `head_length`: 箭头头部的长度。
    - `length_includes_head`: 布尔值，如果为 `True`，则 `dx, dy` 定义的长度包含箭头头部；如果为 `False` (默认)，则 `dx, dy` 定义的是箭身长度，箭头头部会在此基础上延伸。
    - `width`: 箭身（线段部分）的宽度。
    - `lw` (linewidth): 箭身的线宽。
    - `color`: 箭头的颜色。
    - `shape`: 箭头的形状，可以是 `'full'`, `'left'`, `'right'`。
    - `overhang`: 箭头头部相对于箭身末端的悬垂量。

`plt.arrow()` 是一个相对低级的绘图函数，适用于绘制简单的方向指示，或者在已知起点和方向向量时绘制箭头。

> `plt.arrow()` 在底层实际上是绘制一个线段和一个多边形（代表箭头头部）。它根据你提供的 `x, y, dx, dy` 来计算线段的起点和终点，并根据 `head_width`, `head_length` 等参数来计算箭头头部的多边形顶点。这些图形元素作为 `Artist` 对象被添加到当前的 `Axes` 中。

- **与 `annotate()` 的区别**: `plt.arrow()` 绘制的是一个独立的箭头，不附带文本。而 `ax.annotate()` 主要用于为文本添加一个指向特定点的箭头，它提供了更丰富的箭头样式和连接方式。如果你只需要一个简单的方向指示而不需要文本，`plt.arrow()` 更直接。
- **路径可视化**: 在路径规划、流场可视化或展示数据点之间的连接关系时，`plt.arrow()` 非常有用。
- **自定义箭头样式**: 虽然 `plt.arrow()` 的样式参数有限，但可以通过多次调用 `plt.arrow()` 或结合 `plt.plot()` 绘制线段和 `plt.Polygon()` 绘制自定义箭头形状来实现更复杂的箭头。

**在机器学习/深度学习/大模型中的应用场景**：

- **梯度方向可视化**: 在优化算法的教学中，可以在损失函数的等高线上绘制梯度方向的箭头，直观展示梯度下降的路径。
- **特征向量方向**: 在 PCA 或其他特征提取方法中，可以绘制主成分的方向向量。
- **数据流向图**: 在展示数据预处理或模型推理流程时，可以用箭头表示数据从一个模块到另一个模块的流向。
- **注意力权重流**: 在 Transformer 模型的注意力机制可视化中，可以用箭头表示查询、键、值之间的交互方向。

---

```python
# 生成随机的位置点
# 为了结果可以被复现，加随机种子
np.random.seed(42)
loc = np.random.randint(0, 11, (10, 2))

plt.figure(figsize=(10, 10))
plt.plot(loc[:, 0], loc[:, 1], "go", ms=15, label="Data Points")
plt.grid(True, ls="--", alpha=0.7)
plt.title("有箭头的随机路径", fontsize=16)
plt.xlabel("X 轴")
plt.ylabel("Y 轴")
plt.xlim(-1, 11)
plt.ylim(-1, 11)

# 生成随机路径
way = np.arange(len(loc))
np.random.shuffle(way)  # 随机打乱顺序


# 遍历路径，绘制箭头和文本：
for i in range(0, len(way)-1):
    start = loc[way[i]]
    end = loc[way[i+1]]
    # 计算箭头的方向和长度
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    # 绘制箭头
    plt.arrow(start[0], start[1], dx, dy, head_width=0.4, head_length=0.6, lw = 1.5, color="darkblue", 
              length_includes_head=True, shape="full", alpha=0.8)
    
    plt.text(start[0]+0.2, start[1]+0.2, s=str(i), fontsize=14, color="red", ha="center", va="center")
    
    # 如果是最后一个箭头，也要在终点添加文本
    if i == len(way) - 2:
        plt.text(end[0]+0.2, end[1]+0.2, s=str(i+1), fontsize=14, color="red", ha="center", va="center")
        
    
plt.legend()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727223710303.png" alt="image-20250727223710303" style="zoom:50%;" />

**选择题**

1. `plt.arrow(x, y, dx, dy, ...)` 中，`dx` 和 `dy` 表示什么？

   A. 箭头的终点坐标。

   B. 箭头的长度和方向向量。

   C. 箭头的宽度和高度。

   D. 箭头的起点坐标。

   > 答案：B，`dx` 和 `dy` 定义了从起点 `(x, y)` 到终点 `(x+dx, y+dy)` 的向量，即箭头的长度和方向。

2. 如果希望 `plt.arrow()` 绘制的箭头长度 `dx, dy` 包含箭头头部，应该设置哪个参数为 `True`？

   A. `include_head` B. `length_includes_head` C. `head_inclusive` D. `full_length`

   > 答案：B，这个参数控制 `dx, dy` 定义的长度是否包含箭头头部。

**编程题**

1. 创建一个散点图，包含三个点：A(1, 1), B(5, 3), C(2, 6)。
2. 使用 `plt.arrow()` 绘制以下箭头：
    - 从 A 指向 B 的箭头，颜色为蓝色，箭头头部宽度 0.3，长度 0.5。
    - 从 B 指向 C 的箭头，颜色为绿色，箭头头部宽度 0.4，长度 0.6。
    - 从 C 指向 A 的箭头，颜色为红色，箭头头部宽度 0.5，长度 0.7。
    - 确保箭头的线宽为 2。
    - 为每个点添加文本标签 'A', 'B', 'C'。

```python
points = {
    "A": (1, 1),
    "B": (5, 3), 
    "C": (2, 6)
}

plt.figure(figsize=(8, 8))
plt.grid(True, ls = "--", alpha=0.7)
plt.title("点和箭头", fontsize=16)

# 点和文本标签
for label, (x, y) in points.items():
    plt.plot(x, y,"o",  markersize=10, color="k")
    plt.text(x + 0.2, y + 0.2, label, fontsize=14, c="darkblue", ha="left", va="bottom")
    
# 画箭头
# A->B
start_A = points["A"]
end_B = points["B"]
dx_AB = end_B[0] - start_A[0]
# 从 A 指向 B 的箭头，颜色为蓝色，箭头头部宽度 0.3，长度 0.5。
dy_AB = end_B[1] - start_A[1]
plt.arrow(start_A[0], start_A[1], dx_AB, dy_AB, color="blue", lw=2, head_width=0.3, head_length=0.5, length_includes_head=True)
    
# B ->C 
start_B = points["B"]
end_C = points["C"]
dx_BC = end_C[0] - start_B[0]
dy_BC = end_C[1] - start_B[1]
# 从 B 指向 C 的箭头，颜色为绿色，箭头头部宽度 0.4，长度 0.6。
plt.arrow(start_B[0], start_B[1], dx_BC, dy_BC, color="green", lw=2, head_width=0.4, head_length=0.6, length_includes_head=True)


# C ->A
start_C = points["C"]
end_A = points["A"]
dx_CA = end_A[0] - start_C[0]
dy_CA = end_A[1] - start_C[1]
# 从 C 指向 A 的箭头，颜色为红色，箭头头部宽度 0.5，长度 0.7。
plt.arrow(start_C[0], start_C[1], dx_CA, dy_CA, color="red", lw=2, head_width=0.5, head_length=0.7, length_includes_head=True)


# 为每个点添加文本标签 'A', 'B', 'C'。

    
    
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727233549570.png" alt="image-20250727233549570" style="zoom:50%;" />





# 3.注释 (`ax.annotate`)

`ax.annotate(text, xy, xytext, arrowprops=None, **kwargs)` 函数是 Matplotlib 中用于添加带箭头的文本注释的最强大和灵活的方法。它允许你指定文本内容、文本位置、箭头指向的数据点，以及箭头的各种样式。

- `text`: 要显示的注释文本字符串。
- `xy`: 箭头的 **指向点** 坐标（数据坐标系）。这是箭头尖端所指的位置。
- `xytext`: 注释文本的 **位置** 坐标（数据坐标系）。这是文本框的参考点。
- `arrowprops`: 一个字典，用于定义箭头的各种属性。这是 `annotate` 函数的核心，提供了极大的灵活性。常用的键值对包括：
    - `facecolor`: 箭头和文本框的填充颜色。
    - `edgecolor`: 箭头和文本框的边缘颜色。
    - `shrink`: 箭头两端收缩的百分比（占总长）。例如 `shrink=0.05` 表示箭头两端各收缩 5%。
    - `width`: 箭身（线段部分）的宽度。
    - `headwidth`: 箭头头部的宽度。
    - `headlength`: 箭头头部的长度。
    - `connectionstyle`: 箭身连接样式（如 `'arc3'`, `'angle'`, `'bar'`），将在下一节详细讨论。
    - `arrowstyle`: 箭头头部样式（如 `'-|>'`, `'->'`, `'<->'`），将在下一节详细讨论。
    - `alpha`: 箭头的透明度。
    - `patchA`, `patchB`: 可以指定一个 `Patch` 对象作为箭头的起点或终点，使箭头连接到图形元素上。
- `**kwargs`: 其他文本属性，如 `fontsize`, `color`, `ha`, `va`, `bbox` 等，这些与 `ax.text()` 的参数类似。

`annotate` 的强大之处在于它能够连接文本和数据点，并提供高度可定制的箭头样式，使其成为在图表中进行详细说明的首选工具。

> `ax.annotate()` 在底层创建了一个 `Annotation` 对象。这个 `Annotation` 对象是一个复合的 `Artist`，它包含一个 `Text` 对象（用于显示注释文本）和一个 `FancyArrowPatch` 对象（用于绘制箭头）。
>
> `Annotation` 对象会根据 `xy` 和 `xytext` 坐标以及 `arrowprops` 中定义的连接样式，动态地计算箭头的路径和形状。它能够处理不同的坐标系（数据坐标、轴坐标、Figure 坐标），并支持复杂的连接逻辑，如曲线箭头、带角度的箭头等。`shrink` 参数的实现是通过在箭头的起点和终点处留出一定的空白区域，避免箭头直接覆盖数据点或文本。

**坐标系转换**: `xycoords`, `textcoords` 参数可以指定 `xy` 和 `xytext` 的坐标系。

- `'data'` (默认): 数据坐标系。
- `'axes fraction'`: Axes 坐标系中的比例值 (0到1)。
- `'figure fraction'`: Figure 坐标系中的比例值 (0到1)。
- `'offset points'`: 相对于 `xy` 或 `xytext` 的像素偏移量。

**文本框 `bbox`**: 可以为注释文本添加背景框，与 `ax.text()` 类似。

**高级箭头样式**: `arrowprops` 中的 `connectionstyle` 和 `arrowstyle` 提供了非常丰富的箭头连接和头部样式，可以创建直角、弧形、带圆角的箭头等。

**应用场景**:

- **突出异常值**: 在散点图或时间序列图中，用箭头指向并解释异常数据点。
- **解释模型决策**: 在分类边界图中，标注某个区域的分类结果，并用箭头指向该区域。
- **趋势变化点**: 在股票图或经济数据图中，标注重要的转折点或事件发生点。
- **机器学习模型输出分析**: 在回归模型的残差图中，标注残差较大的点；在聚类结果图中，标注每个簇的中心或代表性样本。
- **注意力机制解释**: 在自然语言处理中，可以注释出模型在处理某个词时，注意力机制关注了输入序列中的哪些词。

----

```python
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(0, 5, 0.01)
y = np.cos(2 * np.pi * x)* np.exp(-x / 2)

line, = ax.plot(x, y, lw = 2, color="blue", label="衰减余弦波")
ax.grid(True, ls = "--", alpha=0.7)
ax.set_title("注释", fontsize=16)
ax.set_xlabel("时间")
ax.set_ylabel("值")
ax.set_ylim(-1.2, 1.8)

ax.annotate(
    "在t=0处，有最大值",
    xy=(0, 1),
    xytext=(1, 1.5),
    arrowprops=dict(
        facecolor="black", 
        shrink=0.05,  # 箭头两端收缩百分比
        width=1,
        headwidth = 8,
        headlength=10,
        connectionstyle="arc3, rad=0.2" # 弧形连接样式
    ),
    fontsize=12,
    color="darkred", 
    bbox=dict(boxstyle="round, pad=0.3", fc="lightcoral", ec="red", lw=1, alpha=0.7)
)

ax.annotate(
    "局部最小值",
    xy=(0.5, -0.4),
    xytext=(2, -1.0),
    arrowprops=dict(
        facecolor="green", 
        shrink=0.1,  # 箭头两端收缩百分比
        width=2,
        headwidth = 10,
        headlength=12,
        connectionstyle="angle3, angleA=90,angleB=0" # 弧形连接样式
    ),
    fontsize=12,
    color="darkgreen", 
    bbox=dict(boxstyle="square, pad=0.3", fc="lightgreen", ec="green", lw=1, alpha=0.7)
)

ax.annotate(
    "另外一个局部最大值",
    xy=(2, 0.1),
    xytext=(3.5, 0.8),
    arrowprops=dict(
        arrowstyle="-|>",
        color="purple",
        lw=2,
        connectionstyle="arc3, rad=-0.3" # 弧形连接样式
    ),
    fontsize=12,
    color="darkmagenta", 
    bbox=dict(boxstyle="round4, pad=0.3", fc="plum", ec="purple", lw=1, alpha=0.7)
)

ax.annotate(
    "中点",
    xy=(2.5, y[np.argmin(np.abs(x-2.5))]),
    xytext=(0.5, 1.5),
    arrowprops=dict(
        arrowstyle="simple,tail_width=0.5,head_width=4,head_length=8",
        color="orange", 
        lw=1,
        connectionstyle="arc3,rad=0.2" # 弧形连接样式
    ),
    fontsize=12,
    color="darkorange", 
    bbox=dict(boxstyle="sawtooth, pad=0.3", fc="peachpuff", ec="orange", lw=0.5, alpha=0.7)
)

plt.tight_layout()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250728155843559.png" alt="image-20250728155843559" style="zoom:50%;" />

**选择题**

1. 在 `ax.annotate(text, xy, xytext, ...)` 中，`xy` 和 `xytext` 分别代表什么？

   A. `xy` 是文本位置，`xytext` 是箭头指向点。

   B. `xy` 是箭头指向点，`xytext` 是文本位置。

   C. `xy` 和 `xytext` 都是文本位置。

   D. `xy` 和 `xytext` 都是箭头指向点。

   > 答案：B

# 4.注释箭头连接形状 (Annotation Connection Styles)

`ax.annotate()` 函数的 `arrowprops` 字典中的 `connectionstyle` 和 `arrowstyle` 参数提供了对箭头形状和连接方式的精细控制。

**`connectionstyle`**: 定义箭身（从 `xytext` 到 `xy` 的路径）的形状。它是一个字符串，格式通常是 `'style,param1=value1,param2=value2...'`。

常用样式：

- **`'angle'`**: 弯曲的直角连接。
    - `angleA`, `angleB`: 箭头在 `xytext` 和 `xy` 处的角度（度）。
    - `rad`: 弯曲的半径。
- **`'angle3'`**: 另一种直角连接，更简单。
    - `angleA`, `angleB`: 箭头在 `xytext` 和 `xy` 处的角度（度）。
- **`'arc'`**: 弧形连接。
    - `angleA`, `angleB`: 箭头在 `xytext` 和 `xy` 处的角度（度）。
    - `armA`, `armB`: 箭头在 `xytext` 和 `xy` 处的“臂长”（像素）。
    - `rad`: 弯曲的半径。
- **`'arc3'`**: 另一种简单的弧形连接。
    - `rad`: 弯曲的半径。正值表示逆时针弧，负值表示顺时针弧。
- **`'bar'`**: 垂直或水平的直线连接，带有一个中间的“bar”。
    - `fraction`: bar 的位置（0到1之间，0.5为中间）。
    - `angle`: bar 的角度（度）。

**`arrowstyle`**: 定义箭头的头部和尾部样式。它也是一个字符串，格式通常是 `'style,param1=value1,param2=value2...'`。

常用样式：

- **`'-'`**: 简单直线（无箭头）。
- **`'->'`**: 简单箭头（指向 `xy`）。
- **`'-|>'`**: 带有垂直尾巴的箭头。
- **`'<-'`**: 简单箭头（从 `xy` 指向 `xytext`）。
- **`'<->'`**: 双向箭头。
- **`'simple'`**: 简单的三角形箭头。
    - `head_width`, `head_length`, `tail_width`。
- **`'fancy'`**: 更复杂的箭头，可以控制更多细节。
- **`'wedge'`**: 楔形箭头。

通过组合 `connectionstyle` 和 `arrowstyle`，可以创建出各种各样的箭头注释，以满足不同的可视化需求。

