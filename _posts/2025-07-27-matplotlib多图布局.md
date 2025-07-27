---
layout: post
title: "matplotlib-多图布局"
date: 2025-07-27
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- matplotlib
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>



在数据可视化中，我们经常需要在同一张画布上展示多个相关的图表，以便进行对比分析或展示不同维度的数据。Matplotlib 提供了多种灵活的方式来实现多图布局。

# 1.子视图 (Subplots)

子视图是 Matplotlib 中最常用的一种多图布局方式。它允许你在一个 `Figure` (画布) 对象中创建一个规则的网格，并在每个网格单元中放置一个 `Axes` (坐标轴) 对象，即一个独立的子图。

主要函数是 `plt.subplot(nrows, ncols, index, **kwargs)`。

- `nrows`: 网格的行数。
- `ncols`: 网格的列数。
- `index`: 当前子图在网格中的位置（从1开始计数，从左到右，从上到下）。也可以是一个三位整数，例如 `221` 等同于 `(2, 2, 1)`。
- `**kwargs`: 其他可选参数，如 `projection` (用于创建3D图等)。

`plt.subplot()` 函数会返回一个 `Axes` 对象。通过操作这个 `Axes` 对象，你可以独立地绘制数据、设置标题、标签、背景颜色等。

**`plt.figure()` 的作用**： 在创建子图之前，通常会先创建一个 `Figure` 对象。`plt.figure(figsize=(width, height))` 用于创建一个新的画布，并指定其尺寸。如果不显式创建，Matplotlib 会自动创建一个默认的 Figure。

**`Axes` 对象的属性设置**： 一旦获得 `Axes` 对象（例如 `ax = plt.subplot(...)`），你可以使用其 `set_*` 方法来设置子图的各种属性，例如：

- `ax.set_title('Title')`: 设置子图标题。
- `ax.set_xlabel('X-axis Label')`: 设置 X 轴标签。
- `ax.set_ylabel('Y-axis Label')`: 设置 Y 轴标签。
- `ax.set_facecolor('color')`: 设置子图的背景颜色。
- `ax.set_xlim(min, max)`: 设置 X 轴的显示范围。
- `ax.set_ylim(min, max)`: 设置 Y 轴的显示范围。
- `ax.plot(...)`: 在该子图上绘制数据。

**`plt.sca()` 和 `plt.gca()`**：

- `plt.gca()` (Get Current Axes): 获取当前活动的 `Axes` 对象。如果你没有显式地将绘制操作绑定到某个 `Axes` 对象上（例如直接使用 `plt.plot()`），那么 `plt.gca()` 返回的就是 Matplotlib 认为你正在操作的那个 `Axes`。
- `plt.sca(ax)` (Set Current Axes): 设置指定的 `Axes` 对象为当前活动的 `Axes`。这意味着后续的 `plt.*` 函数（如 `plt.plot()`, `plt.title()` 等）将作用于这个被设置的 `Axes` 对象上，而不是之前活动的那个。

> Matplotlib 的绘图是基于一个严格的面向对象层次结构：
>
> 1. **`Figure` (画布)**：最顶层的容器，可以包含多个 `Axes` 对象。它代表了整个图形窗口。
> 2. **`Axes` (坐标轴/子图)**：是实际进行数据绘图的区域。每个 `Axes` 都有自己的 X 轴、Y 轴、标题、图例等。一个 `Figure` 可以包含一个或多个 `Axes`。
> 3. **`Artist` (艺术对象)**：所有在 `Figure` 或 `Axes` 上绘制的元素都是 `Artist` 对象，例如线条 (`Line2D`)、文本 (`Text`)、刻度 (`Tick`)、图像 (`Image`) 等。
>
> `plt.subplot()` 的原理是：它在当前 `Figure` 中创建一个新的 `Axes` 对象，并将其放置在指定的网格位置上。如果该位置已经存在 `Axes` 对象，`plt.subplot()` 会返回已存在的 `Axes` 对象，并将其设置为当前活动 `Axes`。这种机制使得你可以通过 `plt.subplot()` 轻松地在不同的子图之间切换，并进行绘制。
>
> 当你使用 `ax.plot()` 时，你是在直接调用 `Axes` 对象的方法，这是一种更推荐的面向对象编程方式，因为它明确指定了绘图目标。而 `plt.plot()` 则是通过 `plt.gca()` 获取当前 `Axes` 对象，然后在其上调用 `plot()` 方法。

- **更灵活的创建方式 `plt.subplots()`**: 虽然 `plt.subplot()` 适用于单个子图的创建和定位，但当你需要创建多个子图并希望它们共享轴或进行更统一的管理时，`plt.subplots()` 是更强大的选择。它会一次性创建整个 `Figure` 和 `Axes` 数组。我们将在第三节详细讨论。
- **共享轴**: 在创建子图时，可以使用 `sharex=True` 或 `sharey=True` 参数让多个子图共享同一个 X 轴或 Y 轴。这在比较不同数据集但具有相同刻度范围的图表时非常有用。
- **子图的删除和清除**:
    - `fig.delaxes(ax)`: 从 Figure 中删除一个指定的 Axes 对象。
    - `ax.cla()`: 清除一个 Axes 对象中的所有内容，但不删除 Axes 本身。
    - `plt.clf()`: 清除当前 Figure 中的所有 Axes，但 Figure 仍然存在。
    - `plt.close()`: 关闭指定的 Figure 窗口。

**在机器学习/深度学习/大模型中的应用场景**：

- **模型训练过程可视化**: 在训练机器学习模型时，可以实时绘制多个指标（如训练损失、验证损失、准确率、F1分数等）在不同子图中，方便对比和监控模型的收敛情况。
- **特征分布对比**: 在数据预处理阶段，可以在不同子图中展示不同特征的分布（直方图、KDE图），或者同一特征在不同类别下的分布，以发现数据中的模式和异常。
- **模型预测结果展示**: 对于图像分类或目标检测任务，可以在一个子图中显示原始图像，在另一个子图中显示模型的预测结果（例如，边界框和类别标签），便于直观评估模型性能。
- **超参数调优结果分析**: 绘制不同超参数组合下模型性能的变化曲线，帮助选择最佳超参数。
- **注意力机制可视化**: 在深度学习模型（如 Transformer）中，可以将不同注意力头的权重矩阵可视化为热力图，每个子图代表一个注意力头，从而理解模型如何关注输入的不同部分。

----

```python
x = np.linspace(-np.pi, np.pi + 0.0001, 50)
y = np.sin(x)


plt.figure(figsize=(10, 8))


ax1 = plt.subplot(2, 2, 1)
ax1.plot(x, y, color = "red", ls = "--", label="sin(x)" )
ax1.set_facecolor("green")
ax1.set_title("sin 函数", fontsize=14, color="darkblue")
ax1.legend()
ax1.grid(True, ls = ":", alpha = 0.7)

ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor("lightblue")
ax2.plot(x, -y, color="blue", marker="o", markersize=5, label="-sin(x)")
ax2.set_title("-sin 函数", fontsize=16, color="darkgreen")
ax2.legend()
ax2.grid(True, ls="--", color="k")

ax3 = plt.subplot(2, 1, 2)
plt.sca(ax3)
x_large =np.linspace(-np.pi, np.pi + 0.00001, 200)
plt.plot(x_large, np.sin(x_large * x_large), color="purple", lw=2, label=r"$sin(x^2)$")
plt.title("sin(x^2)函数", fontsize=20, color="darkorange")
plt.legend()
plt.grid()

# 调整子图之间的间距，避免重叠
plt.tight_layout(pad=3)

plt.show()


```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727170304888.png" alt="image-20250727170304888" style="zoom:50%;" />

**选择题**

1. 以下哪个函数用于在 Matplotlib 中创建一个规则网格的子图？

   A. `plt.axes()` B. `plt.figure()` C. `plt.subplot()` D. `plt.subplots()`

   > 答案：C
   >
   > `plt.subplot()` 用于在 Figure 中创建单个子图并将其放置在网格中。`plt.axes()` 用于创建任意位置的子图。`plt.figure()` 用于创建整个画布。`plt.subplots()` 用于一次性创建 Figure 和一组子图。

2. 如果想在 `plt.subplot(3, 2, 4)` 创建的子图上设置标题，以下哪种方式是正确的？

   A. `plt.title('My Title')`

   B. `ax.set_title('My Title')` (假设 `ax` 是 `plt.subplot(3, 2, 4)` 的返回值)

   C. `plt.subplot(3, 2, 4).title('My Title')`

   D. A 和 B 都正确

   > 答案：B。`plt.subplot()` 返回一个 `Axes` 对象。直接操作这个 `Axes` 对象（例如 `ax.set_title()`）是设置其属性的推荐面向对象方法。`plt.title()` 会作用于当前活动的 `Axes`，但如果之前没有显式设置，可能不是你想要的子图。

3. `plt.sca(ax)` 的作用是什么？

   A. 清除 `ax` 子图的所有内容。 B. 获取当前活动的 `Axes` 对象。 C. 将 `ax` 设置为当前活动的 `Axes` 对象。 D. 删除 `ax` 子图。

   > 答案：C，`plt.sca()` 用于将指定的 `Axes` 对象设置为 Matplotlib 的当前活动 `Axes`，这样后续的 `plt.*` 函数就会作用于它。`plt.gca()` 是获取当前活动的 `Axes`。

**编程题**

1. 创建一个 `Figure`，并在其中创建 4 个子图，呈 2x2 的网格布局。
    - 第一个子图：绘制 $$y=x^2$$ 的曲线，颜色为蓝色，标题为 "Square Function"。
    - 第二个子图：绘制 y=cos(x) 的散点图，标记为红色圆圈，标题为 "Cosine Scatter"。
    - 第三个子图：绘制$$ y=e^x$$ 的曲线，线条宽度为 3，标题为 "Exponential Growth"。
    - 第四个子图：绘制 y=log(x) 的曲线（注意 x 范围），背景颜色设置为浅灰色，标题为 "Logarithmic Scale"。
    - 确保每个子图都有 X 和 Y 轴标签

```python
x = np.linspace(0.1, 5+0.0001, 100)
y1 = x ** 2
y2 = np.cos(x)
y3 = np.exp(x)
y4 = np.log(x)

plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 2, 1)
ax1.plot(x, y1, color="blue")
ax1.set_title("Square Function")

ax2 = plt.subplot(2, 2, 2)
ax2.scatter(x, y2, ls="-", color="red", marker="o")
ax2.set_title("Cosine Scatter")

ax3 = plt.subplot(2, 2, 3)
ax3.plot(x, y3, lw=3)
ax3.set_title("Exponential Growth")

ax4 = plt.subplot(2,2,4)
ax4.plot(x, y4)
ax4.set_facecolor("lightgray")
ax4.set_title("Logarithmic Scale")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727171157305.png" alt="image-20250727171157305" style="zoom:50%;" />

# 2.嵌套 (Nested Axes)

嵌套轴域（或称为内嵌子图）允许你在一个现有的 `Axes` 对象内部或 `Figure` 的任意位置创建新的 `Axes` 对象。这不同于规则的网格布局，它提供了更精细的控制，可以用于创建放大图、自定义图例或小型摘要图。

主要有两种创建嵌套轴域的方式：

1. **使用 `plt.axes([left, bottom, width, height], \**kwargs)`**:
    - 这个函数直接在当前 `Figure` 上创建一个新的 `Axes` 对象。
    - 参数 `[left, bottom, width, height]` 是一个列表，表示新 `Axes` 的左下角相对于 `Figure` 左下角的坐标以及其宽度和高度。这些值都是 **Figure 坐标系中的比例值**，范围从 0 到 1。
        - `left`: `Axes` 左边缘的 x 坐标。
        - `bottom`: `Axes` 下边缘的 y 坐标。
        - `width`: `Axes` 的宽度。
        - `height`: `Axes` 的高度。
    - 返回新创建的 `Axes` 对象。
2. **使用 `fig.add_axes([left, bottom, width, height], \**kwargs)`**:
    - 这是 `Figure` 对象的一个方法，作用与 `plt.axes()` 类似，但它明确地将新的 `Axes` 添加到指定的 `Figure` 对象上。
    - 参数含义与 `plt.axes()` 相同。
    - 返回新创建的 `Axes` 对象。

这两种方法的主要区别在于，`plt.axes()` 通常作用于当前活动的 Figure，而 `fig.add_axes()` 明确指定了要添加到的 Figure 对象。在面向对象编程中，`fig.add_axes()` 是更推荐的方式，因为它更清晰地指明了操作对象

> Matplotlib 的 `Figure` 对象是最高层的容器。`Axes` 对象是实际绘图的区域。当你使用 `plt.axes()` 或 `fig.add_axes()` 时，你是在 `Figure` 的二维平面上，根据提供的比例坐标 `[left, bottom, width, height]`，定义并插入一个新的 `Axes` 实例。这个新的 `Axes` 实例拥有自己独立的坐标系、标题、标签等，与 Figure 上的其他 `Axes` 互不干扰，但它们都属于同一个 `Figure`。
>
> 这种方式的灵活性在于，你可以将 `Axes` 放置在 Figure 的任何位置，甚至可以与其他 `Axes` 重叠。Matplotlib 会根据它们的创建顺序（或通过 `set_zorder()` 方法）来决定它们的堆叠顺序。

- **放大局部区域 (Inset Axes)**: 嵌套轴域最常见的用途就是创建局部放大图。你可以在一个主图上绘制整体趋势，然后在一个内嵌的子图中放大主图的某个特定区域，以显示更多细节。
- **自定义图例或信息框**: 你可以创建一个小的嵌套轴域，专门用于放置自定义的图例、文本说明或统计信息，而不是使用 Matplotlib 默认的 `plt.legend()`。
- **多维度信息展示**: 在一个复杂的图表中，可以利用嵌套轴域在主图旁边或内部放置一些辅助性的迷你图，例如直方图、箱线图等，来展示主图数据的一些边缘分布信息。
- **`set_zorder()`**: 当多个 `Axes` 对象重叠时，`set_zorder()` 方法可以控制它们的绘制顺序。数值越大，绘制越靠上。

**在机器学习/深度学习/大模型中的应用场景**：

- **局部损失曲线放大**: 在训练过程中，整体损失曲线可能在后期变化很小，难以观察。可以使用嵌套轴域放大损失曲线的后期部分，以便更清楚地看到微小的收敛细节。
- **特征重要性细节**: 在展示模型特征重要性时，主图可以显示所有特征的重要性，而嵌套轴域可以放大最重要的几个特征，显示其精确数值或排名。
- **混淆矩阵局部放大**: 对于大型多分类任务的混淆矩阵，某些小类别可能在整体图中难以辨认。可以放大混淆矩阵的特定区域，以查看具体类别之间的混淆情况。
- **模型输出细节**: 在生成模型（如 GANs, VAEs）中，主图可以展示生成图像的整体分布或多样性，而嵌套轴域可以放大几张高质量的生成图像，展示其细节。

----

【1】方法1:`plt.axes()`方法：

```python
x = np.linspace(-np.pi, np.pi+0.0001, 250)
y = np.sin(x)
y_zoom = np.sin(x * 5)
fig = plt.figure(figsize=(10, 7))

ax_main = fig.add_subplot(111)
ax_main.plot(x, y, color="blue", lw=2, label="主sin(x)函数")
ax_main.set_title("主SIN函数：sin(x)", fontsize=16)
ax_main.set_xlabel("X轴")
ax_main.set_ylabel("Y轴")
ax_main.grid(True, ls="--", alpha=0.6)
ax_main.legend(loc="upper right")



# 嵌套
# [left, bottom, width, height]
ax_insert1 = plt.axes([0.2, 0.6, 0.25, 0.25])
ax_insert1.plot(x, y_zoom, color="green", label="插入sin(5x)")
ax_insert1.set_title("y_zoom:sin(5*x)")
ax_insert1.set_xlim(-0.5, 0.5)
ax_insert1.set_ylim(-1.1, 1.1)
ax_insert1.tick_params(axis="both", which="major", labelsize=8) # 调整刻度标签大小
ax_insert1.grid(True, linestyle=":", alpha=0.5)
ax_insert1.legend(fontsize=8)
ax_insert1.set_facecolor("#f9f9f9")

# plt.tight_layout(rect=[0, 0, 1, 0.95]) # 会有UserWarning

# 特别注意：在创建嵌套图之前调用了 plt.show()，这会导致主图被渲染并显示，
# 而后续的嵌套图代码没有被正确执行或添加到同一图形中
# 所以要嵌套图创建完之后再调用plt.show()
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727190918955.png" alt="image-20250727190918955" style="zoom:50%;" />

【2】方法2:`fig.add_axes()`

```python
x = np.linspace(-np.pi, np.pi+0.0001, 250)
y = np.sin(x)
y_zoom = np.sin(x * 5)
fig = plt.figure(figsize=(10, 7))

ax_main = fig.add_subplot(111)
ax_main.plot(x, y, color="blue", lw=2, label="主sin(x)函数")
ax_main.set_title("主SIN函数：sin(x)", fontsize=16)
ax_main.set_xlabel("X轴")
ax_main.set_ylabel("Y轴")
ax_main.grid(True, ls="--", alpha=0.6)
ax_main.legend(loc="upper right")



# 嵌套
ax_insert2 = fig.add_axes([0.65, 0.2, 0.25, 0.25])
ax_insert2.plot(x, y**2, color="red", linestyle=":", label="嵌套:y = (sin(x))^2")
ax_insert2.set_title(r"$y=sin^2(x)$")
ax_insert2.set_xlabel("X", fontsize=8)
ax_insert2.set_ylabel("Y", fontsize=8)
ax_insert2.tick_params(axis="both", which="major", labelsize=8)
ax_insert2.grid(True, linestyle=":", alpha=0.5)
ax_insert2.legend(fontsize=8)
ax_insert2.set_facecolor("#fff0f0")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727191425017.png" alt="image-20250727191425017" style="zoom:50%;" />

**选择题**

1. `plt.axes([0.1, 0.1, 0.8, 0.8])` 中的参数 `[0.1, 0.1, 0.8, 0.8]` 表示什么？

   A. 新 Axes 的像素坐标。

   B. 新 Axes 的数据坐标。

   C. 新 Axes 相对于 Figure 的比例坐标：`[左边缘, 下边缘, 宽度, 高度]`。

   D. 新 Axes 的行、列、索引。

   > 答案：C， `plt.axes()` 和 `fig.add_axes()` 的参数都是 Figure 坐标系中的相对比例值，范围是 0 到 1。

2. 以下哪种方法可以用于在一个主图内部创建一个局部放大图？

   A. `plt.subplot()`

   B. `plt.subplots()`

   C. `plt.axes()` 或 `fig.add_axes()`

   D. `plt.grid()`

   > 答案：C，这两种方法允许你在 Figure 的任意指定位置创建 Axes，非常适合创建内嵌的局部放大图。`plt.subplot()` 和 `plt.subplots()` 用于规则网格布局。



**编程题**

1. 创建一个主图，绘制 y=sin(x) 在 $$x \in [0,10\pi]$$ 范围内的曲线。
2. 在主图的右下角创建一个内嵌子图，放大主图在$$ x \in [0,\pi]$$ 范围内的部分，并绘制 y=cos(x) 在该范围内的曲线。
    - 主图的标题为 "Long Sine Wave"。
    - 内嵌子图的标题为 "Zoomed In Cosine"。
    - 内嵌子图的背景颜色设置为淡黄色。

```python
x = np.linspace(0, 10*np.pi + 0.00001, 100)
y = np.sin(x)
fig = plt.figure(figsize=(10, 8))
main_ = fig.add_subplot(111)
main_.plot(x, y, color="red")
main_.set_title("Long Sine Wave")

insert_ = fig.add_axes([0.65, 0.2, 0.23, 0.2])

x_small = np.linspace(0, np.pi + 0.0001, 50)
y_small = np.cos(x_small)
insert_.plot(x_small, y_small, color = "darkblue")
insert_.set_title("Zoomed In Cosine")
insert_.set_facecolor("lightyellow")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727192255153.png" alt="image-20250727192255153" style="zoom:50%;" />

# 3.多图布局分格显示 (Grid Layouts)

本节将深入探讨 Matplotlib 中创建多图网格布局的更高级和更灵活的方法，包括均匀布局和不均匀布局。

## 3.1 均匀布局 (plt.subplots())

`plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)` 是 Matplotlib 中创建均匀网格子图的推荐方法。它一次性创建整个 `Figure` 和所有 `Axes` 对象，并以 NumPy 数组的形式返回 `Axes` 对象，这使得管理多个子图变得非常方便。

- `nrows`, `ncols`: 定义网格的行数和列数。
- `sharex`, `sharey`:
    - `False` (默认): 每个子图有独立的 X/Y 轴。
    - `True`: 所有子图共享同一个 X/Y 轴的刻度范围和标签。当一个子图的轴范围改变时，其他共享轴的子图也会随之改变。
    - `'row'`: 每行的子图共享 X/Y 轴。
    - `'col'`: 每列的子图共享 X/Y 轴。
- `squeeze`: 如果为 `True` (默认)，当 `nrows` 或 `ncols` 为 1 时，返回的 `Axes` 数组会被压缩掉单维度。例如，`plt.subplots(1, 2)` 返回一个一维数组，而 `plt.subplots(1, 1)` 返回单个 `Axes` 对象而不是数组。
- `**fig_kw`: 传递给 `plt.figure()` 的关键字参数，例如 `figsize=(width, height)`。

`plt.subplots()` 返回一个元组 `(fig, axes)`：

- `fig`: 创建的 `Figure` 对象。
- `axes`: 一个 NumPy 数组，包含所有创建的 `Axes` 对象。你可以通过索引（例如 `axes[0, 0]`）或解包（例如 `((ax1, ax2), (ax3, ax4)) = axes`）来访问每个子图。

> `plt.subplots()` 的核心在于它封装了 `Figure` 和 `Axes` 的创建过程，并利用了 `GridSpec`（在内部使用）来自动计算每个子图的位置和大小，以确保它们均匀分布并填充 Figure。当 `sharex` 或 `sharey` 设置为 `True` 时，Matplotlib 会在内部将这些 `Axes` 对象的轴连接起来，使得它们的刻度范围和标签保持同步。这对于比较不同数据集但具有相同物理意义的轴非常有用。

- **灵活的 `Axes` 数组解包**: 对于 `(2, 2)` 的网格，你可以使用 `fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)` 来直接获取每个 `Axes` 对象，使得代码更具可读性。
- **全局设置**: `fig.set_figwidth()` 和 `fig.set_figheight()` 可以用来在创建 `Figure` 后调整其尺寸。
- **紧凑布局 `plt.tight_layout()`**: 在多图布局中，子图的标题、轴标签等可能会重叠。`plt.tight_layout()` 会自动调整子图参数，使之填充整个 Figure 区域，并减少重叠。这是一个非常实用的函数，强烈推荐在多图布局中使用。
- **`GridSpec` 的底层支持**: 尽管 `plt.subplots()` 提供了便捷的均匀布局，但其内部机制依赖于 `GridSpec`。理解 `GridSpec` 对于掌握不均匀布局至关重要。

**在机器学习/深度学习/大模型中的应用场景**：

- **多模型性能对比**: 在一个 `3x3` 的网格中，每个子图可以展示一个不同模型的性能指标（例如，不同分类器在同一数据集上的 ROC 曲线或 PR 曲线），方便横向对比。
- **超参数网格搜索结果**: 如果你进行了超参数的网格搜索，每个子图可以代表一种超参数组合下的模型表现，例如，不同学习率和批大小组合下的训练损失图。
- **数据增强效果展示**: 对于图像数据，可以展示原始图像和经过不同数据增强（旋转、裁剪、翻转等）后的图像，每个子图一张。
- **多任务学习的可视化**: 如果模型同时执行多个任务（如图像分割、目标检测、深度估计），每个子图可以展示一个任务的输出结果。



----

```python
x = np.linspace(0, 2 * np.pi+0.0001, 100)
# fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 8))
fig, axes = plt.subplots(3, 3, figsize=(10, 8))

# axes.shape   # (3, 3)
# fig.set_figwidth(10)
# fig.set_figheight(8)
axes[0, 0].plot(x, np.sin(x), color="blue")
axes[0, 0].set_title("sin(x)")

axes[0, 1].plot(x, np.cos(x), color="green")
axes[0, 1].set_title("cos(x)")

axes[0, 2].plot(x, np.tanh(x), color="red")
axes[0, 2].set_title("tanh(x)")

axes[1, 0].plot(x, np.tan(x), color="orange")
axes[1, 0].set_title("tan(x)")

axes[1, 1].plot(x, np.cosh(x), color="brown")
axes[1, 1].set_title("cosh(x)")

axes[1, 2].plot(x, np.sinh(x), color="purple")
axes[1, 2].set_title("sinh(x)")

axes[2, 0].plot(x, np.sin(x)+np.cos(x), color="cyan")
axes[2, 0].set_title("sin(x)+cos(x)")

axes[2, 1].plot(x, np.sin(x*x)+np.cos(x * x), color="magenta")
axes[2, 1].set_title(r"$sin(x^2)+cos(x^2)$")

axes[2, 2].plot(x, np.sin(x)*np.cos(x), color="darkblue")
axes[2, 2].set_title("sin(x) * cos(x)")

# 为所有子图，添加网格线,和坐标
for ax in axes:
    for a in ax:
        a.grid(True, linestyle=":", alpha=0.8)
        a.set_xlabel("X轴")
        a.set_ylabel("Y轴")

        
plt.tight_layout(pad=2)        
    
plt.show()

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727193828303.png" alt="image-20250727193828303" style="zoom:50%;" />



## 3.2 不均匀分布 (Non-Uniform Layouts)

当需要创建不规则形状或大小的子图时，有几种更高级的方法：

> 总结：
>
> 在 Matplotlib 中，创建子图布局有三种常用方法，适用于不同场景：
>
> 1. `plt.subplot()`：通过 (nrows, ncols, index) 定义网格，index 支持元组（如 (4, 5)）实现简单跨越，适合快速创建规则或轻度不均匀布局。但灵活性有限，难以处理复杂网格。
> 2. `plt.subplot2grid()`：通过 `shape=(nrows, ncols)` 和` loc=(row, col) `指定子图位置，支持 rowspan 和 colspan，直观且适合规则网格中部分子图大小不一的场景，但仍受限于网格规则。
> 3. GridSpec：最强大的方法，使用` gridspec.GridSpec `定义网格，支持切片（如` gs[0, :`]）和 `width_ratios/height_ratios` 自定义子图比例。适合复杂、不规则布局，推荐结合` fig.add_subplot` 使用以实现精确控制。

**方式一: 使用 `plt.subplot()` 结合元组或切片索引**

虽然 `plt.subplot()` 通常用于均匀网格，但它可以通过 `index` 参数的特殊形式来实现简单的不均匀布局。

- `plt.subplot(nrows, ncols, index)`: 当 `index` 是一个元组时，例如 `(row_start, col_start)`，或者一个切片时，例如 `(row_slice, col_slice)`，它会尝试创建一个跨越多行或多列的子图。
- 例如：`plt.subplot(3, 3, (4, 5))` 表示在 3x3 的网格中，从第 4 个位置开始，跨越到第 5 个位置（即占据第 2 行的第 1 和第 2 列）。这种方式的灵活性有限，因为 `index` 仍然是基于线性索引的。

**方式二: 使用 `plt.subplot2grid()`**

`plt.subplot2grid(shape, loc, rowspan=1, colspan=1, **kwargs)` 提供了一种更直观的方式来指定子图的起始位置和跨越的行/列数。

- `shape`: 一个元组 `(nrows, ncols)`，定义了整个网格的行数和列数。
- `loc`: 一个元组 `(row, col)`，指定了当前子图的左上角在网格中的起始位置（从 0 开始计数）。
- `rowspan`: 子图跨越的行数（默认为 1）。
- `colspan`: 子图跨越的列数（默认为 1）。
- 返回新创建的 `Axes` 对象。

这种方法非常适合在规则网格的基础上创建一些跨越多个单元格的子图。

**方式三: 使用 `matplotlib.gridspec.GridSpec`**

`GridSpec` 是 Matplotlib 中最强大和最灵活的布局管理器。它允许你：

1. **定义网格**: `gs = gridspec.GridSpec(nrows, ncols, **kwargs)`，创建一个网格规范对象。
    - `width_ratios`: 一个列表，指定每列的相对宽度比例。
    - `height_ratios`: 一个列表，指定每行的相对高度比例。
2. **通过切片选择区域**: `gs[row_slice, col_slice]` 可以通过 NumPy 风格的切片来选择网格中的一个或多个单元格。
    - `gs[0, :]`: 选中第一行的所有列。
    - `gs[1:, 2]`: 选中从第二行开始到最后一行的第三列。
    - `gs[-1, 0]`: 选中倒数第一行的第一列。
3. **将 `Axes` 添加到指定区域**:
    - `fig.add_subplot(gs[row_slice, col_slice])`: 将新的 `Axes` 对象添加到 `Figure` 的指定 `GridSpec` 区域。这是推荐的面向对象方式。
    - `plt.subplot(gs[row_slice, col_slice])`: 也可以通过 `plt.subplot()` 直接使用 `GridSpec` 对象来创建子图。

`GridSpec` 提供了对子图大小和位置的最高级别控制，甚至可以创建非矩形区域（尽管这通常需要更复杂的组合）。

> 这些不均匀布局方法的原理都是在 `Figure` 层面进行空间划分。
>
> - `plt.subplot()` 的元组/切片索引方式，实际上是 Matplotlib 内部对网格单元进行合并的一种简化表示。
> - `plt.subplot2grid()` 是对 `GridSpec` 更高层次的封装，它提供了一个方便的接口来指定子图的起始点和跨度，内部仍然是基于网格系统进行计算。
> - `GridSpec` 则是直接暴露了底层网格布局的抽象。它并不直接创建 `Axes` 对象，而是定义了一个“蓝图”或“规范”，描述了 Figure 应该如何被划分为子区域。然后，你可以将 `Axes` 对象“放置”到这些预定义的区域中。`GridSpec` 允许你精确控制行高和列宽的比例，这在需要强调某个子图时非常有用。

- **`GridSpecFromSubplotSpec`**: 当你需要在 `GridSpec` 定义的某个大区域内部再进行更细致的网格划分时，可以使用 `gridspec.GridSpecFromSubplotSpec`。
- **灵活的布局调整**: `GridSpec` 允许你创建非常复杂的布局，例如 L 形布局、T 形布局等，通过巧妙地组合切片和 `rowspan`/`colspan`。
- **`fig.add_subplot()` vs `plt.subplot()`**: 再次强调，当使用 `GridSpec` 时，`fig.add_subplot(gs[...])` 是更推荐的面向对象方法，它明确地将子图添加到特定的 Figure 和 GridSpec 区域。
- **应用场景**:
    - **Dashboard 布局**: 创建包含不同类型图表（如主趋势图、统计摘要图、详细分布图）的复杂仪表板。
    - **科学出版物**: 在论文或报告中，需要将多个相关但大小不一的图表组合在一起，以清晰地展示研究结果。
    - **数据探索工具**: 构建交互式数据探索界面时，可以根据用户选择动态调整图表的布局和大小。
    - **机器学习模型诊断**: 例如，在主图显示模型整体性能，旁边放置混淆矩阵、ROC 曲线、特征重要性等辅助图，这些辅助图可能大小不一。

----

【1】方法1:plt.subplot()

```python
x = np.linspace(0,2 * np.pi+0.00001, 200)
plt.figure(figsize=(12, 9))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(x, np.sin(10*x), color="blue")
ax1.set_title("sin(10x)")

ax2 = plt.subplot(3, 3, (4, 5))
ax2.set_facecolor("#e0ffe0")
ax2.plot(x, np.cos(x), color="red")
ax2.set_title("cos(x)")

ax3 = plt.subplot(3, 3, (6, 9))
ax3.plot(x, np.sin(x)+np.cos(x), color="purple")
ax3.set_title("sin(x)+cos(x)")

ax4 = plt.subplot(3, 3, 7)
ax4.plot([1, 3], [2, 4], "o-")

ax5 = plt.subplot(3, 3, 8)
ax5.plot([1, 2, 3], [0, 2, 4], color="orange", marker="s")


plt.tight_layout(pad=1)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727200056039.png" alt="image-20250727200056039" style="zoom:50%;" />

【2】plt.subplot2grid()

```python
x = np.linspace(0, 2* np.pi, 200)
plt.figure(figsize=(12, 9))

ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=3)
ax1.set_facecolor("red")

ax2 = plt.subplot2grid(shape=(3, 3), loc=(1, 0), colspan=2)
ax2.set_facecolor("blue")

ax3 = plt.subplot2grid(shape=(3, 3), loc=(1, 2), rowspan=2)
ax3.set_facecolor("yellow")

ax4 = plt.subplot2grid(shape=(3, 3), loc=(2, 0))
ax4.set_facecolor("purple")

ax5 = plt.subplot2grid((3, 3), (2, 1))
ax5.set_facecolor("gold")
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727200538771.png" alt="image-20250727200538771" style="zoom:50%;" />

【3】方法3:最灵活的方法：matplotlib.gridspec.GridSpec

```python
import matplotlib.gridspec as gridspec

x = np.linspace(0, 2 * np.pi + 0.0001, 200)
plt.figure(figsize=(12, 9))

gs = gridspec.GridSpec(3, 3, width_ratios= [1, 2, 1], height_ratios=[1, 2, 1])
ax1 = plt.subplot(gs[0, :])
ax1.set_facecolor("red")

ax2 = plt.subplot(gs[1, :2])
ax2.set_facecolor("yellow")

ax3 = plt.subplot(gs[1:, 2])
ax3.set_facecolor("k")

ax4 = plt.subplot(gs[2, 0])
ax4.set_facecolor("blue")

ax5 = plt.subplot(gs[2, 1])
ax5.set_facecolor("pink")
```



<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727201357316.png" alt="image-20250727201357316" style="zoom:50%;" />

若写为：

```python
gs = gridspec.GridSpec(3, 3, width_ratios= [1, 1, 1], height_ratios=[1, 1, 1])
```

则效果和前两种的结果一样。

**选择题**

1. 要创建一个 2x2 的均匀子图网格，并希望所有子图共享 X 轴，最简洁的函数调用是：

   A. `fig, axes = plt.subplots(2, 2, sharex=True)`

   B. `plt.subplot(2, 2, 1); plt.subplot(2, 2, 2); ...`

   C. `plt.subplot2grid((2, 2), (0, 0), sharex=True)`

   D. `gridspec.GridSpec(2, 2)`

   > 答案：A，`plt.subplots()` 是创建均匀网格子图的推荐方法，`sharex=True` 参数可以直接实现 X 轴共享。

2. `plt.subplot2grid((4, 4), (1, 1), rowspan=2, colspan=2)` 创建的子图将占据 4x4 网格的哪个区域？

   A. 从第 1 行第 1 列开始，跨越 2 行 2 列。

   B. 从第 2 行第 2 列开始，跨越 2 行 2 列。

   C. 从第 1 行第 1 列开始，占据第 2 行和第 2 列。

   D. 从第 2 行第 2 列开始，占据第 2 行和第 2 列。

   > 答案：B， `loc=(row, col)` 是从 0 开始计数的。所以 `(1, 1)` 对应第二行第二列。`rowspan=2` 意味着它会占据当前行和下一行，`colspan=2` 意味着它会占据当前列和下一列。

3. 以下哪种方法在创建不均匀布局时提供了对行高和列宽比例的直接控制？

   A. `plt.subplot()`

   B. `plt.subplot2grid()`

   C. `matplotlib.gridspec.GridSpec`

   D. `plt.axes()`

   > 答案：C, `GridSpec` 允许通过 `width_ratios` 和 `height_ratios` 参数直接控制行高和列宽的比例，这是其他方法不具备的。

**编程题**

1. 使用 `matplotlib.gridspec.GridSpec` 创建一个复杂的图表布局：
    - 整个 Figure 分为 3 行 3 列。
    - 第一行占据所有 3 列，绘制 y=sin(x)。
    - 第二行分为两部分：
        - 左侧占据第 1 和第 2 列，绘制 y=cos(x)。
        - 右侧占据第 3 列，绘制 y=tan(x)（注意 Y 轴范围）。
    - 第三行分为两部分：
        - 左侧占据第 1 列，绘制一个简单的散点图 `(1,2), (3,4)`。
        - 右侧占据第 2 和第 3 列，绘制 $$y=x^3$$。
    - 为每个子图添加标题，并使用 `plt.tight_layout()` 调整布局。

```python
import matplotlib.gridspec as gridspec

plt.figure(figsize=(10, 8))
x = np.linspace(0, 2 * np.pi + 0.0001, 100)


gs = gridspec.GridSpec(3, 3)

ax1 = plt.subplot(gs[0, :])
ax1.plot(x, np.sin(x), color="red")
ax1.set_title("sin(x)")

ax2_1 = plt.subplot(gs[1, :2])
ax2_1.plot(x, np.cos(x), color="blue")
ax2_1.set_title("cos(x)")

ax2_2 = plt.subplot(gs[1, 2])
ax2_2.plot(x, np.tan(x), color="m")
ax2_2.set_title("tan(x)")

ax3_1 = plt.subplot(gs[2, 0])
ax3_1.scatter([1, 2], [3, 4], color="purple")
ax3_1.set_title("散点图")

ax3_2 = plt.subplot(gs[2, 1:])
ax3_2.plot(x, x**3, color="orange")
ax3_2.set_title(r"$x^3$")

plt.tight_layout(pad=2)
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727210044880.png" alt="image-20250727210044880" style="zoom:50%;" />

# 4.双轴显示 (Twin Axes)

双轴显示允许你在同一个 `Axes` 对象上绘制两组具有不同 Y 轴刻度的数据。这在需要比较两个具有不同量纲或范围的数据系列时非常有用，例如同时绘制温度和降水量随时间的变化。

主要函数是 `ax.twinx()` 和 `ax.twiny()`：

- `ax.twinx()`:
    - 在现有 `Axes` 对象 `ax` 的基础上创建一个新的 `Axes` 对象。
    - 新 `Axes` 对象共享 `ax` 的 X 轴，但拥有独立的 Y 轴。
    - 返回新创建的 `Axes` 对象。
- `ax.twiny()`:
    - 在现有 `Axes` 对象 `ax` 的基础上创建一个新的 `Axes` 对象。
    - 新 `Axes` 对象共享 `ax` 的 Y 轴，但拥有独立的 X 轴。
    - 返回新创建的 `Axes` 对象。

**使用步骤**：

1. 创建一个普通的 `Axes` 对象（例如，通过 `plt.gca()` 或 `fig.add_subplot()`）。
2. 在该 `Axes` 上绘制第一组数据，并设置其 Y 轴标签和刻度颜色。
3. 调用 `ax.twinx()` (或 `ax.twiny()`) 创建第二个 `Axes` 对象。
4. 在新创建的 `Axes` 对象上绘制第二组数据，并设置其 Y 轴标签和刻度颜色。
5. 通常需要为两个 Y 轴使用不同的颜色，以便区分。

> 当调用 `ax.twinx()` 时，Matplotlib 会创建一个新的 `Axes` 实例，并将其放置在与原始 `Axes` 完全相同的位置上。关键在于，这个新的 `Axes` 的 X 轴与原始 `Axes` 的 X 轴是“链接”或“共享”的。这意味着当一个 `Axes` 的 X 轴范围改变时，另一个 `Axes` 的 X 轴也会同步改变。然而，它们的 Y 轴是完全独立的，各自拥有自己的刻度、标签和范围。这种机制使得两个数据集可以共享同一个 X 轴的物理意义，但各自在自己的 Y 轴上进行缩放和显示。

- **三轴或更多轴**: 虽然 `twinx()` 和 `twiny()` 提供了双轴功能，但 Matplotlib 也支持创建更多轴。这通常通过 `ax.secondary_xaxis()` 和 `ax.secondary_yaxis()` 来实现，它们允许你在主轴的另一侧添加次要轴，并可以定义它们与主轴的转换关系。这在需要复杂单位转换或多重刻度时非常有用。
- **颜色和图例管理**: 为了避免混淆，强烈建议为每个 Y 轴及其对应的数据曲线使用不同的颜色。同时，确保图例能够清晰地指示每条曲线对应哪个 Y 轴。
- **潜在的混淆**: 双轴图虽然强大，但如果使用不当，也可能导致图表难以理解。应确保两个 Y 轴的数据之间存在逻辑关联，并且读者能够清楚地辨别哪个曲线对应哪个轴。避免在同一个轴上绘制过多曲线。

**在机器学习/深度学习/大模型中的应用场景**：

- **损失与准确率曲线**: 在模型训练过程中，可以同时绘制训练损失（通常是较大的数值）和训练准确率（通常是 0 到 1 之间的比例）随 epoch 的变化，使用双轴可以清晰地展示它们的趋势。
- **学习率与模型性能**: 绘制学习率调度曲线（学习率随 epoch 变化）和模型在验证集上的性能指标（如 F1 分数），以观察学习率策略对模型收敛的影响。
- **特征值与累计贡献率**: 在主成分分析 (PCA) 中，可以同时绘制每个主成分的特征值（或解释方差）和累计解释方差比例，双轴可以更好地展示两者的关系。
- **资源消耗与性能**: 在训练大型模型时，可以同时监控 GPU 内存使用率/CPU 利用率与模型训练速度/验证损失，以优化资源分配。
- **模型预测概率与实际标签分布**: 在分类任务中，可以绘制模型预测的某个类别的概率分布，同时在另一个 Y 轴上显示该类别实际标签的频率分布。

----

```python
t = np.linspace(0, 10+0.0001, 100)
data1 = np.exp(-0.5*t)+np.sin(3*t)
data2 = t**2 + 5

plt.figure(figsize=(10, 6))
plt.rcParams["font.size"] = 14  # 全局设置字体大小


ax1 = plt.gca()
ax1.plot(x, data1, color="red", lw=2, label="数据1")
ax1.set_xlabel("时间(s)", fontsize=12)
ax1.set_ylabel("exp^(-0.5t)*sin(3t)", color="red", fontsize=12)
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid(True, ls = "--", alpha=0.6)


ax2 = ax1.twinx()
ax2.set_ylabel(r"$t^2$", color="blue", fontsize=12)
ax2.plot(x, data2, color="blue", lw=2, label="数据2")
ax2.tick_params(axis="y", labelcolor="blue")

# 图例需要合并
line1, label1 = ax1.get_legend_handles_labels()
line2, label2 = ax2.get_legend_handles_labels()
plt.legend(line1+line2, label1+label2, loc="upper center", fontsize=12)

plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727213646395.png" alt="image-20250727213646395" style="zoom:50%;" />



**选择题**

1. `ax.twinx()` 的作用是：

   A. 创建一个与 `ax` 共享 Y 轴的新 Axes。

   B. 创建一个与 `ax` 共享 X 轴的新 Axes。

   C. 将 `ax` 的 X 轴和 Y 轴都翻转。

   D. 复制 `ax` 的所有属性到一个新的 Axes。

   > 答案：B，`twinx()` 意味着“孪生 X 轴”，即共享 X 轴。

2. 在双轴图中，为了更好地区分两条曲线，最推荐的做法是：

   A. 增加线条宽度。

   B. 使用不同的线条样式。

   C. 为每个 Y 轴及其对应曲线使用不同的颜色。

   D. 仅在图例中说明。

   > 答案：C。使用不同的颜色是最直观、最有效的区分方法，并且将 Y 轴刻度标签颜色与曲线颜色对应起来，能进一步增强可读性。

**编程题**

1. 绘制一个双轴图，显示以下两组数据随时间 `t` 的变化：
    - `t` 范围从 0 到 5，共 100 个点。
    - 第一组数据：`y1 = 100 * np.exp(-t/2)` (模拟温度下降)。
    - 第二组数据：`y2 = 5 * np.cos(t * 2)` (模拟振动)。
    - 将温度曲线绘制在左侧 Y 轴，颜色为橙色，标签为 "Temperature (°C)"。
    - 将振动曲线绘制在右侧 Y 轴，颜色为绿色，标签为 "Vibration (mm)"。
    -  确保两个 Y 轴的刻度标签颜色与其曲线颜色一致。
    - 添加一个统一的图例。

```python
t = np.linspace(0, 5+0.00001, 100)
data1 = 100*np.exp(-t/2)
data2 = 5*np.cos(2*t)

plt.figure(figsize=(12, 9))

ax1 = plt.subplot(1, 1, 1)
ax1.plot(t, data1, color="orange")
ax1.set_ylabel("Temperature (°C)", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(t, data2, color="green")
ax2.set_ylabel("Vibration (mm)", color="green")
ax2.tick_params(axis="y", labelcolor="green")

line1, label1 = ax1.get_legend_handles_labels()
line2, label2 = ax2.get_legend_handles_labels()

plt.legend(line1+line2, label1+label2, fontsize=14)

plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250727214550968.png" alt="image-20250727214550968" style="zoom:50%;" />









