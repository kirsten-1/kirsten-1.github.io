---
layout: post
title: "matplotlib-基础知识"
date: 2025-07-27
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- matplotlib
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>




为了画图时正确显示中文以及负号：

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False   # 解决负号显示的问题
```

![image.png](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/8a5f6bbafc1248df9cdc95e79fc92520.png)

在数据分析与机器学习中，数据可视化是理解数据、发现模式、验证假设和展示结果的强大工具。一张制作精美的数据图片，可以展示大量的信息，真正做到“一图顶千言”。

Matplotlib 是 Python 最著名的绘图库，它提供了一整套 API，十分适合绘制各种图表，并允许用户精细地控制图表的每一个属性，如字体、标签、范围、颜色、线型等。

**安装 Matplotlib：**

您可以使用 `pip` 命令轻松安装 Matplotlib：

```python
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Matplotlib 是一个 Python 的 2D 绘图库，它能够在交互式环境中生成出版质量级别的图形。通过 Matplotlib 这个标准类库，开发者只需要几行代码就可以实现生成绘图，如折线图、散点图、柱状图、饼图、直方图、组合图等数据分析可视化图表。它不仅功能强大，而且高度可定制，是数据科学家和研究人员的首选工具之一。

---

本部分将介绍 Matplotlib 的基本绘图流程和核心组件，包括图形的创建、坐标轴的设置、网格线、刻度、标签和标题的控制。

# 1.入门图形绘制

在 Matplotlib 中，所有的绘图都是在一个 `Figure`（图形）对象上进行的，而实际的绘图区域则位于 `Axes`（坐标系）对象中。理解 Figure 和 Axes 的概念是掌握 Matplotlib 的关键。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727002308533.png" alt="image-20250727002308533" style="zoom:30%;" />

- **核心概念：Figure 和 Axes**
    - **Figure (图形)**：可以理解为一张白纸或一个窗口，它是所有绘图元素的顶层容器。一个 Figure 可以包含一个或多个 Axes。
    - **Axes (坐标系)**：是实际进行数据绘图的区域。它包含 x 轴、y 轴、刻度、标签、标题等。通常，我们的大部分绘图操作都是在 Axes 对象上进行的。
- **`plt.figure()`：创建图形**
    - **讲解与原理：** `plt.figure()` 用于创建一个新的 Figure 对象。如果您不显式创建，Matplotlib 会在第一次调用绘图函数时自动创建一个默认的 Figure 和 Axes。
    - **参数：**
        - `figsize`: (width, height) 元组，指定图形的宽度和高度（单位为英寸）。
        - `dpi`: 每英寸点数，用于控制图形的分辨率。
        - `facecolor`, `edgecolor`: 背景颜色和边缘颜色。
    - **拓展：**
        - 在 Jupyter Notebook 或其他交互式环境中，每次调用 `plt.figure()` 都会创建一个新的空白图。
        - 通过获取 Figure 对象，可以对其进行更高级的操作，如保存、调整大小等。
    - **应用场景：** 当需要在一个窗口中绘制多个独立的图表，或者需要精确控制图表尺寸和分辨率时。
- **`plt.plot()`：绘制线形图**
    - **讲解与原理：** `plt.plot()` 是 Matplotlib 中最基本的绘图函数之一，用于绘制 2D 线形图。它接受 x 坐标和 y 坐标作为输入，将这些点连接起来形成线。
    - **原理：** `plot` 函数将输入的 x, y 数据点转换为屏幕坐标，并根据指定的线型、颜色等属性绘制像素。
    - **参数：**
        - `x`, `y`: 数据点的 x 和 y 坐标。
        - `color` / `c`: 线的颜色。
        - `linestyle` / `ls`: 线的样式（如 '`-`', '`--`', '`:`', '`-.`'）。
        - `linewidth` / `lw`: 线的宽度。
        - `marker`: 数据点的标记样式（如 '`o`', '`s`', '`^`', '`*`'）。
        - `alpha`: 线的透明度（0.0 到 1.0）。
    - **拓展：**
        - **绘制多条线：** 可以多次调用 `plt.plot()` 在同一个 Axes 上绘制多条线。
        - **快捷参数：** 可以将颜色、线型、点型组合成一个字符串参数，如 `'bo--'` 表示蓝色圆点虚线。
    - **应用场景：** 展示时间序列数据趋势、函数曲线、数据点之间的关系等。
- **`plt.grid()`：设置网格线**
    - **讲解与原理：** 网格线可以帮助用户更精确地读取图表上的数值。`plt.grid()` 用于在坐标系中添加网格线。
    - **原理：** Matplotlib 会根据当前的刻度位置自动生成网格线。
    - **参数：**
        - `b`: 布尔值，是否显示网格线（已废弃，推荐使用 `True`/`False`）。
        - `linestyle` / `ls`: 网格线的样式。
        - `color` / `c`: 网格线的颜色。
        - `alpha`: 网格线的透明度。
        - `axis`: 指定网格线应用于哪个轴 ('x', 'y', 'both')。
    - **拓展：**
        - 可以单独控制 x 轴或 y 轴的网格线。
        - 在数据密度较高或需要精确读数的图表中，网格线非常有用。
    - **应用场景：** 科学绘图、数据分析报告等需要精确数值参考的场景。
- **`plt.axis()` / `plt.xlim()` / `plt.ylim()`：设置坐标轴范围**
    - **讲解与原理：** Matplotlib 会根据数据自动调整坐标轴的显示范围。但有时我们需要手动设置这些范围，以突出特定区域或保持多图之间的一致性。
    - **`plt.axis([xmin, xmax, ymin, ymax])`：** 同时设置 x 和 y 轴的范围。
    - **`plt.xlim([xmin, xmax])`：** 仅设置 x 轴的范围。
    - **`plt.ylim([ymin, ymax])`：** 仅设置 y 轴的范围。
    - **原理：** 这些函数会修改当前 Axes 对象的 x 和 y 轴视图限制。
    - **拓展：**
        - **自动调整：** 如果不设置，Matplotlib 会自动选择合适的范围。
        - **反转轴：** 可以通过设置 `xmax < xmin` 或 `ymax < ymin` 来反转坐标轴。
    - **应用场景：** 放大图表中的特定区域、统一多个子图的坐标轴范围以便比较、排除异常值对视图的影响。

-----

```python
x = np.linspace(-4, 2 * np.pi, 100) # 100个点
y_sin = np.sin(x)
y_cos = np.cos(x)

# 调整图形尺寸
plt.figure(figsize=(9, 6)) # 单位：英寸

# 绘制线形图
plt.plot(x, y_sin, color="blue", linestyle="-")
plt.plot(x, y_cos, color="red", linestyle="--")

# 网格线
plt.grid(linestyle="--", color="grey", alpha=0.4, axis="both")

# 设置坐标轴
plt.axis([-4, 4, -1.2, 1.2])

# 添加图例
# plt.legend()

# 添加标题和轴标签
plt.title("正弦波和余弦波")
plt.xlabel("X轴")
plt.ylabel("Y轴")

#  自动调整子图参数，填充整个图像区域
plt.tight_layout()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727123625261.png" alt="image-20250727123625261" style="zoom:50%;" />

**选择题：**

1. 在 Matplotlib 中，以下哪个是实际进行数据绘图的区域？ A) `Figure` B) `Canvas` C) `Axes` D) `Plot`

> 答案：C，`Figure` 是顶层容器，`Axes` 是实际绘制数据（包含坐标轴、刻度等）的区域。

2. 要在一个 `plt.plot()` 调用中同时设置线的颜色为红色、线型为虚线、点型为圆形，以下哪个参数组合是正确的？

   A) `color='red', linestyle='--', marker='o'`

   B) `'ro--'`

   C) `color='red', ls='--', marker='circle'`

   D) A 和 B 都正确

> 答案：D，A 是使用独立参数设置，B 是使用快捷字符串参数。两者都正确。



**编程题：**

1. 绘制函数 $$y=x^2 在 $$x in `[−5,5]` 范围内的线形图。设置图表尺寸为 8x5 英寸，并添加绿色虚线网格线。将 x 轴范围设置为 `[−6,6]`，y 轴范围设置为 `[0,30]`。

```python
# np.set_printoptions(suppress=True)
x = np.linspace(-6, 6, 200)
y = pow(x, 2)
plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.grid("g--")
plt.axis([-6, 6, 0, 30])
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727124208685.png" alt="image-20250727124208685" style="zoom:50%;" />

使用 `np.linspace` 生成 x 值，计算 y 值。`plt.figure` 设置尺寸，`plt.plot` 绘制曲线。`plt.grid` 添加网格线并设置样式。`plt.xlim` 和 `plt.ylim` 设置坐标轴范围(或者用`plt.axis([xmin, xmax, ymin, ymax])`)。

# 2.坐标轴刻度、标签、标题

图表的刻度、标签和标题是传达图表信息的重要组成部分。它们使得图表更具可读性和专业性。

- **`plt.xticks()` / `plt.yticks()`：设置 x 轴和 y 轴刻度**

    - **讲解与原理：** 这些函数用于手动指定坐标轴上刻度线的位置。默认情况下，Matplotlib 会自动选择刻度位置，但有时我们需要在特定位置显示刻度，例如在数学函数图中显示 pi 的倍数。
    - **参数：**
        - `ticks`: 一个列表或数组，指定刻度线的位置。
        - `labels`: 一个列表或数组，指定每个刻度线对应的标签文本。如果未提供，则使用 `ticks` 的值作为标签。
        - `fontsize`: 标签的字体大小。
        - `rotation`: 标签的旋转角度。
        - `ha` / `horizontalalignment`: 标签的水平对齐方式。
        - `color`, `fontweight`, `fontfamily` 等字体属性。
    - **拓展：**
        - **LaTeX 语法：** 在标签中使用 `r'$\frac{\pi}{2}$'` 这样的原始字符串和 LaTeX 语法，可以渲染出漂亮的数学公式。需要确保 Matplotlib 配置支持 LaTeX（通常默认支持，但复杂公式可能需要安装 TeX 发行版）。
        - **刻度格式化：** 可以使用 `matplotlib.ticker` 模块进行更复杂的刻度格式化，例如百分比、货币等。
    - **应用场景：** 数学函数绘图、时间序列图中每年的开始、自定义分类轴等。

- **`plt.xlabel()` / `plt.ylabel()` / `plt.title()`：设置坐标轴标签和标题**

    - **讲解与原理：** 这些函数用于为 x 轴、y 轴和整个图表添加描述性文本。

    - **参数：**

        - `label`: 标签文本。
        - `fontsize`: 字体大小。
        - `color`: 字体颜色。
        - `fontweight`: 字体粗细。
        - `rotation`: 标签旋转角度（`ylabel` 默认垂直）。
        - `horizontalalignment` / `ha`: 水平对齐方式。
        - `verticalalignment` / `va`: 垂直对齐方式。

    - **拓展：**

        - **中文显示：** 默认情况下，Matplotlib 可能无法正确显示中文。需要通过 `plt.rcParams` 设置字体。

          ```python
          from matplotlib.font_manager import FontManager
          # 获取电脑上的字体库 (可选，用于查看可用字体)
          # fm = FontManager()
          # mat_fonts = set(f.name for f in fm.ttflist)
          # print(mat_fonts)
          
          plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体，例如 'SimHei' (黑体) 或 'Songti SC' (宋体)
          plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
          ```

        - **子图标题：** 对于子图，可以使用 `ax.set_title()` 来设置每个子图的标题。

    - **应用场景：** 任何需要清晰解释图表内容的场景，是图表自解释性的关键。

----

```python
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(9, 6))
plt.plot(x, y, label="正弦函数")

# 设置x轴y轴刻度
# plt.xticks(np.arange(0, 7, (np.pi)/2))
# plt.yticks([-1, 0, 1])
plt.xticks(ticks=np.arange(0, 7, (np.pi)/ 2), labels=[0, r"$\frac{\pi}{2}$", r"${\pi}$",r"$\frac{3\pi}{2}$",r"${2\pi}$"], fontsize=18, fontweight="normal", color="darkgreen")
plt.yticks(ticks=[-1, 0, 1], labels=["最小值", 0, "最大值"], fontsize=16, ha="right", color="blue")
# 设置x轴y轴坐标轴标签的标题
plt.xlabel("角度(弧度)", fontsize=18, color="purple")
plt.ylabel("函数值", rotation=0, ha="right", fontstyle="normal", fontsize=18, color="orange")
plt.title("正弦波曲线图", fontsize=22, fontweight = "bold", color="darkblue")

plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727130110792.png" alt="image-20250727130110792" style="zoom:50%;" />

**选择题：**

1. 要在 Matplotlib 图表中正确显示中文，通常需要修改哪个 `rcParams` 参数？

   A) `plt.rcParams['figure.figsize']`

   B) `plt.rcParams['font.sans-serif']`

   C) `plt.rcParams['axes.grid']`

   D) `plt.rcParams['lines.linewidth']`

> 答案：B，`plt.rcParams['font.sans-serif']` 用于设置无衬线字体，通常用于显示中文。

2. 以下哪个选项可以用于在 Matplotlib 刻度标签中显示数学公式，例如 alpha？

A) `labels=['alpha']` B) `labels=['\alpha']` C) `labels=[r'$\alpha$']` D) `labels=['$\alpha$']`

> 答案：C，需要使用原始字符串 `r''` 和美元符号 `$$` 来包裹 LaTeX 语法。

**编程题：**

1. 绘制函数 $$y=e^{−x}sin(2 \pi x)$$ 在 x in `[0,4]` 范围内的线形图。
    - 设置图表标题为“衰减正弦波”。
    - x 轴标签为“时间 (秒)”，y 轴标签为“振幅”。
    - x 轴刻度显示为 `0, 1, 2, 3, 4`。
    - y 轴刻度显示为 `-1, 0, 1`。
    - 确保中文显示正常。

```python
x = np.linspace(0, 4, 100)
y = np.exp(-x) * np.sin(2 * np.pi * x)

plt.plot(x, y)
plt.title("衰减正弦波")
plt.xlabel("时间（秒）")
plt.ylabel("振幅")
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([-1, 0, 1])
# 确保中文显示正常
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727131100467.png" alt="image-20250727131100467" style="zoom:50%;" />

# 3.图例

当图表中包含多条曲线或多个数据系列时，图例（Legend）是必不可少的，它帮助读者区分和理解每个数据系列的含义。

- **`plt.legend()`：添加图例**
    - **讲解与原理：** `plt.legend()` 用于在图表中添加图例。它会根据 `plt.plot()` 或其他绘图函数中设置的 `label` 参数来生成图例项。
    - **原理：** Matplotlib 会收集所有带有 `label` 参数的绘图对象，并在图例框中显示这些标签及其对应的颜色/线型/点型。
    - **参数：**
        - `labels`: 一个字符串列表，显式指定图例的标签。如果 `plot` 函数已经设置了 `label`，则无需再次指定。
        - `loc`: 图例的位置。可以是字符串（如 'upper right', 'lower left', 'center', 'best' 等）或整数（1-10）。'best' 会自动选择最佳位置。
        - `ncol`: 图例的列数。
        - `fontsize`: 图例文本的字体大小。
        - `bbox_to_anchor`: 一个元组 `(x, y, width, height)`，用于将图例放置在 Axes 外部的任意位置。这在图例会遮挡数据时非常有用。
    - **拓展：**
        - **自定义图例句柄：** 对于更复杂的图例（例如，一个图例项代表多个绘图对象），可以使用 `matplotlib.patches` 创建自定义的图例句柄。
        - **图例标题：** 可以通过 `title` 参数为图例添加标题。
    - **应用场景：** 比较不同模型性能、展示不同类别数据分布、多变量时间序列图等。

----

```python
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(9, 6))

plt.plot(x, y_sin, color="blue", label="正弦波")
plt.plot(x, y_cos, color="red", label="余弦波")

#  frameon=True显示图例边框
#   ncol = 1图例显示为1列
# facecolor="lightyellow" 图例背景颜色
# edgecolor="gray" 图例边框颜色
plt.legend(fontsize=14, 
           loc="lower left", 
           ncol = 1, 
           title="函数类型", 
           frameon=True, 
           shadow=True, 
           facecolor="lightyellow",
          edgecolor="gray")


plt.title("正弦波与余弦波")
plt.xlabel("X轴")
plt.ylabel("Y轴")
plt.grid(linestyle=":", alpha=0.5)
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727132603483.png" alt="image-20250727132603483" style="zoom:50%;" />

关于参数`bbox_to_anchor`

```python
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(9, 6))

plt.plot(x, y_sin, color="blue", label="正弦波")
plt.plot(x, y_cos, color="red", label="余弦波")

#  borderaxespad=0  轴与图例之间的间距
plt.legend(fontsize=14, 
           loc="lower left", 
           bbox_to_anchor=(0, 1.02, 0.4, 0.2),  # （x, y, width, height）相对figure的坐标
           mode="expand", # 展开图例以填充bbox_to_anchor定义的宽度
           ncol = 2, 
           title="函数类型", 
           borderaxespad=0)


plt.title("正弦波与余弦波")
plt.xlabel("X轴")
plt.ylabel("Y轴")
plt.grid(linestyle=":", alpha=0.5)
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727132940486.png" alt="image-20250727132940486" style="zoom:50%;" />

**选择题：**

1. 要在 Matplotlib 图表中为多条曲线添加图例，以下哪个参数在 `plt.plot()` 中是必需的？

   A) `color` B) `linestyle` C) `label` D) `legend`

   > 答案：C，`label` 参数用于为每条曲线指定图例文本。`plt.legend()` 会收集这些 `label` 来生成图例。

2. 要将图例放置在图表的右侧中央，并且图例项横向排列成两列，应该如何设置 `plt.legend()` 的参数？

   A) `loc='center right', ncol=2`

   B) `loc='right', ncol=2`

   C) `loc='center', bbox_to_anchor=(1.05, 0.5), ncol=2`

   D) `loc='right', mode='expand', ncol=2`

   > 答案：A，其实B也对。
   >
   > 其他选项：
   >
   > C：
   >
   > <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727133307925.png" alt="image-20250727133307925" style="zoom:50%;" />
   >
   > D：
   >
   > <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727133337784.png" alt="image-20250727133337784" style="zoom:50%;" />

   **编程题：**

    1. 绘制函数 $$y_1=sin(x) $$和$$ y_2=cos(x) $$在$$ x in [0,2 \pi]$$ 范围内的线形图。
        - 为 $$y_1$$ 曲线添加标签“正弦函数”，为 $$y_2$$ 曲线添加标签“余弦函数”。
        - 将图例放置在图表的左上角。
        - 图例字体大小设置为 12。
        - 图例边框显示，并有轻微阴影。

   ```python
   x = np.linspace(0, 2 * np.pi, 100)
   y_1 = np.sin(x)
   y_2 = np.cos(x)
   plt.plot(x, y_1, label="正弦函数")
   plt.plot(x, y_2, label="余弦函数")
   plt.legend(loc="upper left", fontsize=12, frameon=True, shadow=True )
   plt.show()
   ```

   <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727134354940.png" alt="image-20250727134354940" style="zoom:50%;" />



# 4.脊柱移动

在 Matplotlib 中，“脊柱”（Spines）指的是围绕数据区域的线，它们代表了坐标轴的边界。默认情况下，图表有四条脊柱（上、下、左、右）。通过移动或隐藏脊柱，可以创建更具艺术感或特定用途的图表。

- **`plt.gca()`：获取当前 Axes**
    - **讲解与原理：** `plt.gca()` 是 "Get Current Axes" 的缩写。它返回当前活动的 `Axes` 对象。在 Matplotlib 的状态机接口（`pyplot` 模块）中，很多函数（如 `plt.plot`, `plt.title`）都是对当前 Axes 进行操作。但如果需要对 Axes 对象进行更精细的控制（例如，移动脊柱），就需要先获取到这个 Axes 对象。
    - **原理：** Matplotlib 维护一个当前的 Figure 和 Axes 栈。`plt.gca()` 返回栈顶的 Axes 对象。
    - **拓展：**
        - **面向对象接口：** 在更复杂的绘图中，通常直接创建 Figure 和 Axes 对象（例如 `fig, ax = plt.subplots()`），然后直接在 `ax` 对象上调用方法（如 `ax.plot()`, `ax.set_title()`），这样更清晰和可控。
    - **应用场景：** 需要对 Axes 级别属性进行修改时。
- **`ax.spines`：访问脊柱对象**
    - **讲解与原理：** `ax.spines` 是一个字典状的对象，可以通过键（'left', 'right', 'top', 'bottom'）访问到对应的 `Spine` 对象。每个 `Spine` 对象都有自己的属性和方法来控制其外观和位置。
    - **`set_color()`：设置脊柱颜色**
        - **讲解与原理：** 用于改变脊柱的颜色。设置为 'none' 或 'white' 可以使其不可见。
    - **`set_position()`：设置脊柱位置**
        - **讲解与原理：** 用于改变脊柱相对于其默认位置的偏移。
        - **参数：**
            - `'outward'`: 将脊柱向外移动指定点数。
            - `'axes'`: 将脊柱放置在 Axes 坐标系中的相对位置（0.0 到 1.0）。
            - `'data'`: 将脊柱放置在数据坐标系中的特定值。这对于将坐标轴原点移动到 (0,0) 非常有用。
    - **原理：** `Spine` 对象是 `matplotlib.spines.Spine` 类的实例，它封装了对坐标轴边界线的操作逻辑。
    - **拓展：**
        - **隐藏脊柱：** 将 `set_color()` 设置为透明或背景色可以隐藏脊柱。
        - **自定义外观：** 可以设置脊柱的线宽、线型等。
    - **应用场景：** 绘制数学函数图（将坐标轴移到原点）、美化图表、创建无边框图表等。

----

```python
x = np.linspace(0, 2 * np.pi+0.001, 256) # 更多点，使得曲线更加平滑
y_1 = np.sin(x)
y_2 = np.cos(x)

plt.figure(figsize=(9, 6))
plt.plot(x, y_1, label="正弦波", color="blue", linewidth=2)
plt.plot(x, y_2, label="余弦波", color="red", linewidth=2)

plt.legend()
ax = plt.gca()

ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

ax.spines["bottom"].set_position(("data", 0))  
ax.spines["left"].set_position(("data", 0))  

# 刻度线调整，不仅可以用plt.xticks或者plt.yticks，也可以用ax.set_xticks
ax.set_xticks([-np.pi, -np.pi / 2, 0,np.pi / 6, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], 
              labels=[r"$-\pi$",  r"$\frac{-\pi}{2}$", 0, r"$\frac{\pi}{6}$",r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{2\pi}{2}$", r"$2\pi$"],
             fontsize=16)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727151716767.png" alt="image-20250727151716767" style="zoom:50%;" />

**选择题：**

1. 要获取当前 Matplotlib 图形中的 `Axes` 对象，以便对其进行更细粒度的控制，应该使用哪个函数？

   A) `plt.figure()` B) `plt.plot()` C) `plt.gca()` D) `plt.show()`

   > 答案：C，`plt.gca()` 用于获取当前活动的 `Axes` 对象

2. 要将 x 轴的脊柱移动到 y 轴数据值为 0 的位置，以下哪个 `set_position` 参数设置是正确的？

   A) `ax.spines['bottom'].set_position(('axes', 0))`

   B) `ax.spines['bottom'].set_position(('data', 0))`

   C) `ax.spines['left'].set_position(('data', 0))`

   D) `ax.spines['bottom'].set_position(0)`

   > 答案：B， `('data', 0)` 表示将脊柱放置在数据坐标系中 y 值为 0 的位置。

**编程题：**

1. 绘制一条从 (−2,−2) 到 (2,2) 的直线。
    - 隐藏图表的顶部和右侧脊柱。
    - 将底部和左侧脊柱移动到数据原点 (0,0)。
    - 设置 x 轴和 y 轴的刻度为 `-2, 0, 2`。
    - 添加标题“直线图 (脊柱在原点)”。

```python
x = np.linspace(-2, 2+0.0001, 200)
y = x
plt.plot(x, y)
# 隐藏图表的顶部和右侧脊柱。
ax = plt.gca()
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
# 将底部和左侧脊柱移动到数据原点 (0,0)
ax.spines["bottom"].set_position(("data", 0))
ax.spines["left"].set_position(("data", 0))
# 设置 x 轴和 y 轴的刻度为 -2, 0, 2
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])
# 添加标题“直线图 (脊柱在原点)”
ax.set_title("直线图(脊柱在原点)")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727152301418.png" alt="image-20250727152301418" style="zoom:50%;" />



# 5.图片保存

生成高质量的图表后，通常需要将其保存为图片文件，以便在报告、演示文稿或论文中使用。Matplotlib 提供了灵活的保存功能。

- **`plt.savefig()`：保存图形**
    - **讲解与原理：** `plt.savefig()` 用于将当前 Figure 保存到文件中。它支持多种文件格式（如 PNG, JPG, PDF, SVG 等），并提供了丰富的参数来控制输出质量和布局。
    - **原理：** Matplotlib 会根据当前的 Figure 对象和其包含的 Axes 对象，将其渲染成像素或矢量图形，并写入指定的文件。
    - **参数：**
        - `fname`: 文件名或文件路径，包括文件扩展名（如 'my_plot.png', 'report/figure1.pdf'）。
        - `dpi`: 每英寸点数，控制图像的分辨率。对于位图格式（如 PNG, JPG），更高的 DPI 意味着更清晰的图像。
        - `facecolor`, `edgecolor`: 控制 Figure 的背景颜色和边框颜色。
        - `bbox_inches`: 控制保存的区域。
            - `'tight'`: 自动调整图表周围的空白区域，使其尽可能紧凑，防止标签或标题被裁剪。这是最常用的设置。
            - `None`: 使用 Figure 的原始大小。
        - `pad_inches`: 当 `bbox_inches='tight'` 时，额外添加的边距（英寸）。
        - `transparent`: 是否使 Figure 背景透明。
    - **拓展：**
        - **矢量图 vs. 位图：**
            - **位图 (Raster Graphics)**：如 PNG, JPG。由像素组成，放大后会失真。适合网页和屏幕显示。
            - **矢量图 (Vector Graphics)**：如 PDF, SVG。由数学公式描述，放大后不会失真。适合出版物和需要高分辨率打印的场景。
        - **`plt.tight_layout()`：** 在保存前调用 `plt.tight_layout()` 是一个好习惯，它会自动调整子图参数，使之填充整个图像区域，并避免标签重叠或被裁剪。
    - **应用场景：** 将分析结果导出为图片用于报告、论文、网页展示等。
- **`ax.set_facecolor()`：设置 Axes 背景颜色**
    - **讲解与原理：** `ax.set_facecolor()` 用于设置当前 Axes（绘图区域）的背景颜色。这与 `plt.figure(facecolor=...)` 不同，后者设置的是整个 Figure 的背景颜色。
    - **原理：** 直接修改 Axes 对象的背景属性。
    - **拓展：**
        - 可以用于突出图表中的特定区域，或与数据颜色方案协调。
    - **应用场景：** 美化图表、强调数据区域。



-----

```python
x = np.linspace(0, 2 * np.pi + 0.00001, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 7), linewidth = 4, edgecolor="lightblue")
plt.plot(x, y_sin, label = "正弦波", linewidth=2, color = "red")
plt.plot(x, y_cos, label = "余弦波", linewidth=2, color = "k")

# 获取视图
ax = plt.gca()
# 设置视图的背景颜色
ax.set_facecolor("lightgreen")

# 图例
plt.legend(fontsize = 14, loc = "lower left", ncol = 2, title="函数曲线")

plt.title("正弦波与余弦波曲线图", fontsize=18)

plt.xlabel("X轴", fontsize=14)
plt.ylabel("Y轴", fontsize=14, rotation=0, labelpad=10)
plt.grid(linestyle=":", alpha = 0.6)

plt.tight_layout()

plt.savefig("./基础练习.pdf", dpi=150, facecolor="violet", edgecolor="navy", bbox_inches="tight", pad_inches = 0.1)

plt.show()
```

jupyter 中：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727154621903.png" alt="image-20250727154621903" style="zoom:50%;" />

保存的PDF：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727154635878.png" alt="image-20250727154635878" style="zoom:50%;" />

**选择题：**

1. 要在保存 Matplotlib 图表时，确保所有标签和标题都不会被裁剪，应该将 `plt.savefig()` 的哪个参数设置为 `'tight'`？

   A) `dpi` B) `facecolor` C) `bbox_inches` D) `transparent`

   > 答案：C，`bbox_inches='tight'` 会自动调整保存区域，以包含所有图表元素。

2. 以下哪种文件格式在保存 Matplotlib 图表时，放大后不会失真？

   A) JPEG B) PNG C) GIF D) PDF

   > 答案：D，PDF 是一种矢量图格式，放大后不会失真。JPEG, PNG, GIF 都是位图格式。
   >
   > - **位图 (Raster Graphics)**：如 PNG, JPG。由像素组成，放大后会失真。适合网页和屏幕显示。
   > - **矢量图 (Vector Graphics)**：如 PDF, SVG。由数学公式描述，放大后不会失真。适合出版物和需要高分辨率打印的场景。

**编程题：**

1. 绘制一个简单的散点图，包含 50 个随机点。
    - 设置图表标题为“随机散点图”。
    - 将 Axes 背景颜色设置为浅蓝色。
    - 将图形保存为 `scatter_plot.pdf` 文件，DPI 设置为 300，并确保保存完整。

```python
x_data = np.random.rand(50) * 10
y_data = np.random.rand(50) * 10

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color="purple", s = 100, alpha = 0.7, edgecolor="black")
ax = plt.gca()
ax.set_facecolor("lightblue")
ax.set_title("随机散点图", fontsize=20)
plt.xlabel("X值", fontsize=14)
plt.ylabel("Y值", fontsize=14)
plt.grid(linestyle="--", alpha=0.6)

# 保存
plt.savefig("./scatter_plot.pdf", dpi=300, bbox_inches = "tight")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727155234050.png" alt="image-20250727155234050" style="zoom:50%;" />









