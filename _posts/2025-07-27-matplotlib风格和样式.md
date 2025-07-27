---
layout: post
title: "matplotlib-风格和样式"
date: 2025-07-27
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- matplotlib
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>


Matplotlib 提供了极其丰富的选项来控制图表中每一个元素的风格和样式，从而创建出美观且信息量大的可视化作品。



# 1.颜色、线形、点形、线宽、透明度

这些是控制线形图外观最基本的参数，理解它们的用法是创建清晰图表的基础。

- **`plt.plot()` 参数详解：**
    - **`color` / `c`：颜色**
        - **讲解与原理：** 可以使用颜色名称（如 'red', 'blue', 'green', 'black'/'k' 等）、HTML 颜色代码（如 '#FF00EE'）、RGB 元组（如 `(0.2, 0.7, 0.2)`）。
        - **原理：** Matplotlib 内部将这些颜色表示转换为 RGB 或 RGBA 值，然后用于渲染线条或标记。
        - **拓展：**
            - **颜色映射 (Colormaps)：** 对于表示连续数值的颜色，可以使用颜色映射（如 `cmap='viridis'`），通常与 `scatter` 或 `imshow` 结合使用。
            - **颜色循环：** Matplotlib 有一个默认的颜色循环，当绘制多条线时不指定颜色时，会自动使用不同的颜色。
    - **`linestyle` / `ls`：线型**
        - **讲解与原理：** 控制线的样式，如实线、虚线、点线等。
        - **常用值：** `'-'` (实线), `'--'` (虚线), `'-.'` (点划线), `':'` (点线), `'None'` (无线条)。
        - **原理：** 通过在绘制时跳过或重复像素来模拟不同的线型。
    - **`marker`：点型**
        - **讲解与原理：** 控制数据点上显示的标记样式。
        - **常用值：** `'o'` (圆圈), `'s'` (正方形), `'^'` (三角形), `'*'` (星形), `'p'` (五边形), `'+'` (加号), `'x'` (叉号) 等。
        - **原理：** 在每个数据点的位置绘制指定的形状。
    - **`linewidth` / `lw`：线宽**
        - **讲解与原理：** 控制线的粗细。值越大，线越粗。
        - **原理：** 增加绘制线条的像素宽度。
    - **`alpha`：透明度**
        - **讲解与原理：** 控制绘图元素的透明度，值范围从 0.0（完全透明）到 1.0（完全不透明）。
        - **原理：** 在渲染时，将绘图元素的颜色与背景颜色进行混合。
        - **拓展：**
            - 在绘制大量重叠数据点（如散点图）或多条曲线时，设置透明度可以帮助观察数据密度或区分重叠区域。
    - **参数连用：**
        - **讲解与原理：** Matplotlib 允许将颜色、线型、点型组合成一个字符串作为 `plot` 函数的第三个参数（在 x, y 之后）。例如 `'bo--'` 表示蓝色圆点虚线。
        - **原理：** 这是一个方便的快捷方式，内部会解析这个字符串并设置相应的参数。
        - **注意：** 这种快捷方式只能设置颜色、线型、点型，其他参数（如 `linewidth`, `alpha`）仍需单独指定。
    - **应用场景：** 精细控制图表外观，使图表更具表现力和可读性，例如区分不同类别的数据、突出重要趋势等。

-----

```python
x = np.linspace(0, 2 * np.pi + 0.0001, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure(figsize=(10, 7))

plt.plot(x, y1, color = "indigo", ls = "-.", marker="p", label="正弦波(点划线，五边形)")
plt.plot(x, y2, color = "#FF00EE", ls="--", marker = "o", label = "余弦波（虚线，圆形）")
plt.plot(x, y1 + y2, color=(0.2, 0.7, 0.2), marker = "*", ls=":", label="和（点线，星型）")
plt.plot(x, y1 + 2 * y2, linewidth=3, alpha=0.4, color="orange", label="加倍余弦(粗线， 半透明)")
plt.plot(x, 2 * y1-y2, "bo--", label="2倍正弦减余弦(圆点虚线)")
plt.legend()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727160230551.png" alt="image-20250727160230551" style="zoom:50%;" />

**选择题：**

1. 要将一条线的颜色设置为红色，线宽设置为 2，透明度设置为 0.5，以下哪个 `plt.plot()` 调用是正确的？

   A) `plt.plot(x, y, 'r', lw=2, alpha=0.5)`

   B) `plt.plot(x, y, color='red', linewidth=2, alpha=0.5)`

   C) `plt.plot(x, y, 'r', linewidth=2, alpha=0.5)`

   D) 以上所有选项都正确

   > 答案：D，A, B, C 都是正确的用法。A 使用了快捷颜色参数，B 和 C 使用了完整的参数名。

2. 以下哪个选项能够使图表中的重叠数据点（例如在散点图中）更容易区分数据密度？

   A) 增加 `linewidth` B) 改变 `marker` 样式 C) 调整 `alpha` 参数 D) 使用 `linestyle`

   > 答案：C，`alpha` 参数控制透明度，当多个点重叠时，透明度较低的点会叠加颜色，从而显示出数据密度。

**编程题：**

1. 绘制函数 y=cos(x) 在 x in `[0,4 pi] `范围内的线形图。
    - 线的颜色设置为深绿色。
    - 线型为点线。
    - 点型为星形。
    - 线宽为 1.5。
    - 透明度为 0.8。
    - 添加标题“余弦函数曲线图”。

```python
x = np.linspace(0, 4 * np.pi + 0.0001, 100)
y = np.cos(x)

plt.plot(x, y, color="darkgreen", linestyle=":", marker="*", linewidth = 1.5, alpha=0.8)
plt.title("余弦函数曲线图")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727160758105.png" alt="image-20250727160758105" style="zoom:50%;" />

# 2.更多属性设置

除了基本的线和点属性，Matplotlib 还提供了更多精细的控制选项，特别是针对标记（marker）和刻度（ticks）的样式。

- **标记 (Marker) 的更多属性：**
    - **`markerfacecolor`：点的填充颜色**
        - **讲解与原理：** 当 `marker` 参数指定了点型时，`markerfacecolor` 用于设置标记内部的填充颜色。
    - **`markersize`：点的大小**
        - **讲解与原理：** 控制标记的尺寸。
    - **`markeredgecolor`：点边缘颜色**
        - **讲解与原理：** 控制标记边缘的颜色。
    - **`markeredgewidth`：点边缘宽度**
        - **讲解与原理：** 控制标记边缘的宽度。
    - **原理：** 这些参数直接修改了 Matplotlib 内部用于绘制标记的图形对象的属性。
    - **拓展：**
        - 结合 `alpha` 可以创建半透明的标记，用于显示数据密度。
        - 在散点图中，这些参数可以用于区分不同类别的数据点，或者通过大小、颜色等编码更多信息。
    - **应用场景：** 突出图表中的关键数据点、创建自定义的散点图样式、在时间序列中标记特定事件点等。
- **刻度 (Ticks) 的大小设置：**
    - **`plt.xticks(size=...)` / `plt.yticks(size=...)`：设置刻度标签大小**
        - **讲解与原理：** 除了在 `plt.xticks()` 和 `plt.yticks()` 中通过 `fontsize` 参数设置刻度标签的字体大小外，也可以使用 `size` 参数，两者效果相同。
        - **原理：** 直接修改刻度标签文本对象的字体大小属性。
    - **拓展：**
        - 可以进一步控制刻度线的长度、粗细、方向等，通过 `ax.tick_params()` 方法。
    - **应用场景：** 确保刻度标签在不同分辨率的图表中清晰可读，或根据图表整体风格进行调整。

```python
def f(x):
    return np.exp(-x) * np.cos(2 * np.pi * x)

x = np.linspace(0, 5 + 0.00001, 50)
y = f(x)

plt.figure(figsize=(9, 6))


plt.plot(x, y, color="purple", marker = "o", ls = "--", lw = 2, alpha = 0.6, 
         markerfacecolor="red", markersize = 10, markeredgecolor="green", markeredgewidth=2, label="衰减余弦波")
plt.legend()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727161351473.png" alt="image-20250727161351473" style="zoom:50%;" />

**选择题：**

1. 要在 `plt.plot()` 中设置标记内部的填充颜色，应该使用哪个参数？

   A) `color` B) `markercolor` C) `markerfacecolor` D) `edgecolor`

   > 答案：C，`markerfacecolor` 用于设置标记的填充颜色。

2. 以下哪个参数用于设置 `plt.xticks()` 或 `plt.yticks()` 中刻度标签的字体大小？

   A) `size` B) `fontsize` C) `font_size` D) A 和 B 都正确

   > 答案：D，`size` 和 `fontsize` 都可以用于设置刻度标签的字体大小，两者是等效的。

**编程题：**

1. 绘制函数 $$y=x^3−3x$$ 在 $$x \in [−3,3]$$ 范围内的线形图。
    - 线的颜色设置为蓝色。
    - 线型为实线。
    - 点型为菱形 (`'D'`)。
    - 点的大小为 8。
    - 点的填充颜色为黄色。
    - 点的边缘颜色为黑色，边缘宽度为 1。
    - x 轴和 y 轴刻度标签字体大小设置为 12。
    - 添加标题“函数 $$y=x^3−3x$$”。

```python
x = np.linspace(-3, 3, 100)
y = x ** 3 + 3 * x

plt.figure(figsize=(9, 6))
plt.plot(x, y, color="blue", ls="-", marker="D", markersize=8, markerfacecolor="yellow", markeredgecolor="k",
        markeredgewidth=1)

plt.xticks(fontsize=12)
plt.yticks(size=12)

plt.title(r"$y=x^3-3x$")

plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727161859083.png" alt="image-20250727161859083" style="zoom:50%;" />







