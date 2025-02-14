---
layout: post
title: "numpy(11)玩转 Pygame"
subtitle: "第 11 章 玩转 Pygame"
date: 2025-02-14
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 人工智能AI基础
---


<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>



前10章以及其他补充已经整理如下：

[第1章NumPy入门](https://kirsten-1.github.io/2025/01/27/numpy(1)_%E5%85%A5%E9%97%A8/)

[第2章NumPy基础](https://kirsten-1.github.io/2025/02/04/numpy(2)_numpy%E5%9F%BA%E7%A1%80/)

[第3章常用函数](https://kirsten-1.github.io/2025/02/06/Numpy(3)%E5%B8%B8%E7%94%A8%E5%87%BD%E6%95%B0/)

[【补充1】-广播与广播机制](https://kirsten-1.github.io/2025/02/04/Numpy%E7%9A%84%E5%B9%BF%E6%92%AD%E5%92%8C%E5%B9%BF%E6%92%AD%E6%9C%BA%E5%88%B6/)

[第4章便捷函数](https://kirsten-1.github.io/2025/02/07/NumPy(4)%E4%BE%BF%E6%8D%B7%E5%87%BD%E6%95%B0/)

[第5章矩阵和通用函数](https://kirsten-1.github.io/2025/02/09/NumPy(5)%E7%9F%A9%E9%98%B5%E5%92%8C%E9%80%9A%E7%94%A8%E5%87%BD%E6%95%B0/)

[第6章深入学习NumPy模块](https://kirsten-1.github.io/2025/02/09/NumPy(6)%E6%B7%B1%E5%85%A5%E5%AD%A6%E4%B9%A0NumPy%E6%A8%A1%E5%9D%97/)

[第7章专用函数](https://kirsten-1.github.io/2025/02/10/NumPy(7)%E4%B8%93%E7%94%A8%E5%87%BD%E6%95%B0/)

[第8章质量控制](https://kirsten-1.github.io/2025/02/12/NumPy(8)%E8%B4%A8%E9%87%8F%E6%8E%A7%E5%88%B6/)

[第9章使用Matplotlib绘图](https://kirsten-1.github.io/2025/02/12/NumPy(9)%E4%BD%BF%E7%94%A8Matplotlib%E7%BB%98%E5%9B%BE/)

[【补充2】-读图片-jupyter notebook-三种方式+图像简单操作](https://kirsten-1.github.io/2025/02/11/jupyter-notebook-%E8%AF%BB%E5%9B%BE%E7%89%87/)

[第10章NumPy 的扩展：SciPy](https://kirsten-1.github.io/2025/02/12/NumPy(10)NumPy%E7%9A%84%E6%89%A9%E5%B1%95-SciPy/)

----

本章写给需要使用NumPy和Pygame快速并且简易地进行游戏制作的开发者。基本的游戏开发经验对于阅读本章内容有帮助，但并不是必需的。

本章涵盖以下内容：

- Pygame基础；
- Matplotlib集成；
- 屏幕像素矩阵；
- 人工智能；
- 动画；
- OpenGL。

---

# 11.1 Pygame

Pygame最初是由Pete Shinners编写的一套Python架构。顾名思义，Pygame可以用于制作电子游戏。自2004年起，Pygame成为GPL（General Public License，通用公共许可证）下的开源免费软件，这意味着你可以使用它制作任何类型的游戏。Pygame基于SDL（Simple DirectMedia Layer，简易直控媒体层）。SDL是一套C语言架构，可用于在各种操作系统中（包括Linux、Mac OS X和Windows）访问图形、声音、键盘以及其他输入设备。

# 11.2 动手实践：安装 Pygame

在本节教程中，我们将安装Pygame。Pygame基本上可以与所有版本的Python兼容。不过在编写本书的时候，和Python 3仍有一些兼容问题，但这些问题很可能不久就会被修复。请完成如下步骤安装Pygame。

(1) 根据你所使用的操作系统，选择一种方式安装Pygame。

- Debian和Ubuntu Pygame可以在Debian软件库中找到： http://packages.qa.debian.org/p/pygame.html。
- Windows 根据所使用的Python版本，我们可以从Pygame的网站上（http://www.pygame.org/download.shtml）下载合适的二进制安装包。
- Mac Pygame在Mac OS X 10.3及以上版本的二进制安装包也可以在这里下载： http://www.pygame.org/download.shtml。

> 注：前10章我在jupyter notebook中运行代码案例。这一章`Pygame` 需要一个主事件循环来处理用户输入、更新屏幕等。然而，Jupyter Notebook 是基于 cell 执行的，而每个 cell 都是相互独立的，这与 `Pygame` 的事件循环模型不兼容。
>
> `Pygame` 需要一个持久的窗口显示，而 Jupyter Notebook 通常在单个输出框内显示图形，这对于动态更新（如游戏中的帧更新）来说比较困难。
>
> 综上，我在pycharm中书写这一章的示例代码。

在pycharm中安装依赖pygame

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214144041449.png" alt="image-20250214144041449" style="zoom:50%;" />

我也不清楚需不需要下面的库（pygame所依赖的），暂时先安装上：

```python
所依赖的库:sdl hg  
  这两个我没有安装成功：portmidizero  pyPortmidi  
```

验证安装:安装完成后，可以编写一个简单的Pygame程序来验证安装是否成功。创建一个Python文件，命名为`test_pygame.py`，并将以下代码复制到文件中:

```python
import pygame

# 初始化Pygame
pygame.init()

# 创建游戏窗口
window = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Pygame测试")

# 游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 绘制背景
    window.fill((255, 255, 255))

    # 刷新窗口
    pygame.display.update()

# 退出游戏
pygame.quit()

```

相应目录下运行`python test_pygame.py`

控制台输出：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214144942287.png" alt="image-20250214144942287" style="zoom:50%;" />

并且有一个弹窗出现：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214144956999.png" alt="image-20250214144956999" style="zoom:50%;" />

这表明已成功安装Pygame并准备好开始游戏开发了！

# 11.3 Hello World

我们将制作一个简单的游戏，并在本章后续内容中加以改进。按照程序设计类书籍的传统，我们将从一个Hello World示例程序开始。

# 11.4 动手实践：制作简单游戏

值得注意的是，所有的动作都会在所谓的游戏主循环中发生，以及使用font模块来呈现文本。在这个程序中，我们将利用Pygame的Surface对象进行绘图，并处理一个退出事件。请完成如下步骤。

(1) 首先，导入所需要的Pygame模块。如果Pygame已经正确安装，将不会有任何报错；否则，请返回安装教程。

```python
import pygame, sys 
from pygame.locals import *
```

(2) 我们将初始化Pygame，创建一块`400 × 300`像素大小的显示区域，并将窗口标题设置为Hello World!。

```python
pygame.init() 
screen = pygame.display.set_mode((400, 300)) 
pygame.display.set_caption('Hello World!') 
```

(3) 游戏通常会有一个主循环一直运行，直到退出事件的发生。在本例中，我们仅仅在坐标`(100, 100)`处设置一个Hello World文本标签，文本的字体大小为19，颜色为红色。

```python
while True:
    sysFont = pygame.font.SysFont("None", 19)
    rendered = sysFont.render ('Hello World', 0, (255, 100, 100))
    screen.blit(rendered, (100, 100))
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214145507694.png" alt="image-20250214145507694" style="zoom:50%;" />

刚才做了些什么 : 在本节教程中，虽然看起来内容不多，但其实我们已经学习了很多。我们将出现过的函数总结在下面的表格中。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214145617262.png" alt="image-20250214145617262" style="zoom:50%;" />

完整代码如下：

```python
import pygame, sys
from pygame.locals import *

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption('Hello World!')

while True:
    sysFont = pygame.font.SysFont("None", 19)
    rendered = sysFont.render ('Hello World', 0, (255, 100, 100))
    screen.blit(rendered, (100, 100))
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
```

# 11.5 动画

大部分游戏，即使是最“静态”的那些，也有一定程度的动画部分。从一个程序员的角度来看，动画只不过是不同的时间在不同地点显示对象，从而模拟对象的移动。

Pygame提供Clock对象，用于控制每秒钟绘图的帧数。这可以保证动画与CPU的快慢无关。

# 11.6 动手实践：使用 NumPy 和 Pygame 制作动画对象

我们将载入一个图像并使用NumPy定义一条沿屏幕的顺时针路径。请完成如下步骤。

(1) 创建一个Pygame的Clock对象，如下所示：

```python
clock = pygame.time.Clock()
```

(2) 和本书配套的源代码文件一起，有一张头部的图片。我们将载入这张图片，并使之在屏幕上移动。

> 注：`head.jpg`可以从https://github.com/sundaygeek/numpy-beginner-guide/blob/master/ch11code/head.jpg 下载。

```python
img = pygame.image.load('head.jpg')
```

(3) 我们将定义一些数组来储存动画中图片的位置坐标。既然对象可以被移动，那么应该有四个方向——上、下、左、右。每一个方向上都有40个等距的步长。我们将各方向上的值全部初始化为0。

```python
steps = np.linspace(20, 360, 40).astype(int) 
right = np.zeros((2, len(steps))) 
down = np.zeros((2, len(steps))) 
left = np.zeros((2, len(steps))) 
up = np.zeros((2, len(steps))) 
```

(4) 设置图片的位置坐标是一件很烦琐的事情。不过，有一个小技巧可以用上——`[::-1]`可以获得倒序的数组元素。

```python
right[0] = steps
right[1] = 20
down[0] = 360
down[1] = steps
left[0] = steps[::-1]
left[1] = 360
up[0] = 20
up[1] = steps[::-1] 
```

(5) 四个方向的路径可以连接在一起，但需要先用T操作符对数组进行转置操作，使得它们以正确的方式对齐。

```python
pos = np.concatenate((right.T, down.T, left.T, up.T)) 
```

(6) 在主循环中，我们设置时钟周期为每秒30帧：

```python
clock.tick(30) 
```

动画截图如下：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214153128684.png" alt="image-20250214153128684" style="zoom:50%;" />

完整代码如下：

```python
import pygame, sys
from pygame.locals import *
import numpy as np

pygame.init()
# 创建一个Pygame的Clock对象 用于控制每秒钟绘图的帧数。这可以保证动画与CPU的快慢无关。
clock = pygame.time.Clock()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('Animating Objects')
# 载入图片
img = pygame.image.load('head.jpg')
steps = np.linspace(20, 360, 40).astype(int)
right = np.zeros((2, len(steps)))
down = np.zeros((2, len(steps)))
left = np.zeros((2, len(steps)))
up = np.zeros((2, len(steps)))

right[0] = steps
right[1] = 20
down[0] = 360
down[1] = steps
left[0] = steps[::-1]
left[1] = 360
up[0] = 20
up[1] = steps[::-1]

pos = np.concatenate((right.T, down.T, left.T, up.T))
i = 0

while True:
    # 清屏
    screen.fill((255, 255, 255))
    if i >= len(pos):
        i = 0
    screen.blit(img, pos[i])
    i += 1
    for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
    pygame.display.update()
    clock.tick(30)
```

刚才做了些什么 :在本节教程中我们学习了一点关于动画的内容，其中最重要的就是时钟的概念。我们将使用到的新函数总结在下面的表格中。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214153217535.png" alt="image-20250214153217535" style="zoom:50%;" />

# 11.7 Matplotlib

我们在第9章中学习过Matplotlib，这是一个可以便捷绘图的开源工具库。我们可以在Pygame中集成Matplotlib，绘制各种各样的图像。

# 11.8 动手实践：在 Pygame 中使用 Matplotlib

在本节教程中，我们将使用前一节教程中的位置坐标并为其绘制图像。请完成如下步骤。

(1) 使用非交互式的后台：为了在Pygame中集成Matplotlib，我们需要使用一个非交互式的后台，否则Matplotlib会默认显示一个GUI窗口。我们将引入Matplotlib主模块并调用use函数。该函数必须在引入Matplotlib主模块后并引入其他Matplotlib模块前立即调用。

```python
import matplotlib as mpl

mpl.use("Agg") 
```

(2) 非交互式绘图可以在Matplotlib画布（canvas）上完成。创建画布需要引入模块、创建图像和子图。我们将指定图像大小为`3 × 3`英寸。更多细节请参阅本节末尾的代码。

```python
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_agg as agg

fig = plt.figure(figsize=[3, 3])  
ax = fig.add_subplot(111)  
canvas = agg.FigureCanvasAgg(fig) 
```

(3) 在非交互模式下绘图比在默认模式下稍复杂一点。由于我们要反复多次绘图，因此有必要将绘图代码组织成一个函数。图像最终应绘制在画布上，这使得我们的步骤变得复杂了一些。在本例的最后，你可以找到这些函数更为详细的说明。

```python
def plot(data):
    ax.plot(data)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "RGB") 
```

> 注：原书有个问题。在较新的版本的 `Matplotlib` 中，`tostring_rgb()` 已被移除，改为 `tostring_argb()` 或 `tostring()`。
>
> 因此，可以修改为使用 `tostring_argb()` 来代替 `tostring_rgb()`。
>
> 但是，`raw_data` 的字节长度与 `size`（图像尺寸）不匹配。可能是由于使用 `tostring_argb()` 获取的字节数据与 `pygame.image.fromstring()` 期望的格式不一致。为了正确处理图像，必须确保图像数据的字节数正确，并且能够正确地传递给 `pygame.image.fromstring()`。

```python
def plot(data):
    ax.plot(data)  # 绘制数据
    canvas.draw()
    renderer = canvas.get_renderer()
    # 使用 tostring_argb() 代替 tostring_rgb()
    raw_data = renderer.tostring_argb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "ARGB")  # 使用 ARGB 格式

```

大致效果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214164127104.png" alt="image-20250214164127104" style="zoom:50%;" />

完整代码：(和书上不太一样，因为我进行了调试，原书某方法已经过时了)

```python
import pygame, sys
from pygame.locals import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

mpl.use("Agg")  # 使用 Agg 后端

# 创建 matplotlib 绘图
fig = plt.figure(figsize=[2, 2])
ax = fig.add_subplot(111)
canvas = agg.FigureCanvasAgg(fig)


def plot(data):
    ax.plot(data)  # 绘制数据
    canvas.draw()
    renderer = canvas.get_renderer()
    # 使用 tostring_argb() 代替 tostring_rgb()
    raw_data = renderer.tostring_argb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "ARGB")  # 使用 ARGB 格式


pygame.init()
clock = pygame.time.Clock()

# 设置显示窗口
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption('Animating Objects')

# 加载图片
img = pygame.image.load('head.jpg')

steps = np.linspace(20, 360, 40).astype(int)
right = np.zeros((2, len(steps)))
down = np.zeros((2, len(steps)))
left = np.zeros((2, len(steps)))
up = np.zeros((2, len(steps)))

right[0] = steps
right[1] = 20
down[0] = 360
down[1] = steps
left[0] = steps[::-1]
left[1] = 360
up[0] = 20
up[1] = steps[::-1]

# 计算路径
pos = np.concatenate((right.T, down.T, left.T, up.T))
i = 0

# 初始化 history
history = np.array([])

# 绘制初始图像
surf = plot(history)

# 主循环
while True:
    # 清空屏幕
    screen.fill((255, 255, 255))

    if i >= len(pos):
        i = 0
        surf = plot(history)  # 更新绘图

    # 绘制图片
    screen.blit(img, pos[i])

    # 更新历史记录
    history = np.append(history, pos[i])

    # 绘制 matplotlib 图形
    screen.blit(surf, (100, 100))  # 在屏幕上绘制 matplotlib 图形

    # 更新帧数
    i += 1

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    clock.tick(30)  # 控制帧率
```

刚才做了些什么 ：下表给出了绘图相关函数的说明。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214164419072.png" alt="image-20250214164419072" style="zoom:50%;" />

# 11.9 屏幕像素

Pygame的surfarray模块可以处理`PygameSurface`对象和NumPy数组之间的转换。你或许还记得，NumPy可以快速、高效地处理大规模数组。

# 11.10 动手实践：访问屏幕像素

在本节教程中，我们将平铺一张小图片以填充游戏界面。请完成如下步骤。

(1) array2d函数将像素存入一个二维数组。还有相似的函数，将像素存入三维数组。我们将avatar头像图片的像素存入数组：

```python
# 将加载的图像转换为一个2D数组，数组中的每个元素表示像素的灰度值
pixels = pygame.surfarray.array2d(img)
```

(2) 我们使用shape属性获取像素数组pixels的形状，并据此创建游戏界面。游戏界面的长和宽都将是像素数组的7倍大小。

```python
# 获取图像的尺寸，并计算将图像放大后的尺寸
X = pixels.shape[0] * 7  # 图像的宽度乘以 7
Y = pixels.shape[1] * 7  # 图像的高度乘以 7

# 创建一个新的pygame窗口，并设置窗口尺寸为放大的图像尺寸
screen = pygame.display.set_mode((X, Y))
```

(3) 使用tile函数可以轻松平铺图片。由于颜色是定义为整数的，像素数据需要被转换成 整数。

```python
# 使用 np.tile() 函数将图像数组放大 7 倍，得到新的数组
# np.tile() 会重复像素数据，创建一个更大的数组
new_pixels = np.tile(pixels, (7, 7)).astype(int)
```

(4) surfarray模块中有一个专用函数`blit_array`，可以将数组中的像素呈现在屏幕上。

```python
# 将新的像素数据渲染到屏幕上
pygame.surfarray.blit_array(screen, new_pixels)
```



效果如下图所示。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214164852621.png" alt="image-20250214164852621" style="zoom:50%;" />

平铺图片的完整代码如下：

```python
import pygame, sys
from pygame.locals import *
import numpy as np

# 初始化pygame库
pygame.init()

# 加载图片 'head.jpg'
img = pygame.image.load('head.jpg')

# 将加载的图像转换为一个2D数组，数组中的每个元素表示像素的灰度值
pixels = pygame.surfarray.array2d(img)

# 获取图像的尺寸，并计算将图像放大后的尺寸
X = pixels.shape[0] * 7  # 图像的宽度乘以 7
Y = pixels.shape[1] * 7  # 图像的高度乘以 7

# 创建一个新的pygame窗口，并设置窗口尺寸为放大的图像尺寸
screen = pygame.display.set_mode((X, Y))

# 设置窗口的标题
pygame.display.set_caption('Surfarray Demo')

# 使用 np.tile() 函数将图像数组放大 7 倍，得到新的数组
# np.tile() 会重复像素数据，创建一个更大的数组
new_pixels = np.tile(pixels, (7, 7)).astype(int)

# 进入主循环，直到用户关闭窗口
while True:
    # 用白色填充整个屏幕
    screen.fill((255, 255, 255))

    # 将新的像素数据渲染到屏幕上
    pygame.surfarray.blit_array(screen, new_pixels)

    # 处理所有的事件
    for event in pygame.event.get():
        # 如果用户点击关闭按钮，退出程序
        if event.type == QUIT:
            pygame.quit()  # 退出pygame
            sys.exit()  # 退出程序

    # 更新显示
    pygame.display.update()

```

刚才做了些什么 :  下面的表格给出了新函数及其属性的简单说明。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214165206440.png" alt="image-20250214165206440" style="zoom:50%;" />

# 11.11 人工智能

在游戏中，我们通常需要模拟一些智能行为。`scikit-learn`项目旨在提供机器学习的API，我最喜欢的是其出色的文档。我们可以使用操作系统的包管理器来安装`scikit-learn`，这取决于你所使用的操作系统是否支持，但应该是最为简便的安装方式。Windows用户可以直接从项目网站上下载安装包。

在Debian和Ubuntu上，该项目名为python-sklearn。在MacPorts命名为py26-scikits-learn和py27-scikits-learn。我们也可以从源代码安装或使用`easy_install`工具，还有第三方发行版如Python(x, y)、Enthought和NetBSD。

> 以上说的是原书那个版本的下载，下面是新版本的（适用于现在）。

在菜单栏中，选择 `文件` > `设置`（在 macOS 上为 `PyCharm` > `首选项`），然后导航到 `项目：您的项目名` > `Python 解释器`。

**安装 scikit-learn**：点击窗口右上角的 `+` 按钮，在弹出的搜索框中输入 `scikit-learn`，选择相应的版本，然后点击 `安装`。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214165553767.png" alt="image-20250214165553767" style="zoom:50%;" />

**验证安装**：安装完成后，您可以在代码中添加以下内容来验证 scikit-learn 是否已正确安装：

```python
import sklearn
print(sklearn.__version__)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214170459859.png" alt="image-20250214170459859" style="zoom:50%;" />

# 11.12 动手实践：数据点聚类

我们将随机生成一些数据点并对它们进行聚类，也就是将相近的点放到同一个聚类中。这只是`scikit-learn`提供的众多技术之一。聚类是一种机器学习算法，即依据相似度对数据点进行分组。随后，我们将计算一个关联矩阵。关联矩阵即包含关联值的矩阵，如点与点之间的距离。最后，我们将使用`scikit-learn`中的`AffinityPropagation`类对数据点进行聚类。请完成如下步骤。

(1) 我们将在`400 × 400`像素的方块内随机生成30个坐标点：

```python
positions = np.random.randint(0, 400, size=(30, 2)) 
```

(2) 我们将使用欧氏距离（Euclidean distance）来初始化关联矩阵。

```python
positions_norms = np.sum(positions ** 2, axis=1)
S = - positions_norms[:, np.newaxis] - positions_norms[np.newaxis, :] + 2 * np.dot(positions, positions.T) 
```

(3) 将前一步的结果提供给AffinityPropagation类。该类将为每一个数据点标记合适的聚类编号。

```python
aff_pro = sklearn.cluster.AffinityPropagation().fit(S)
labels = aff_pro.labels_ 
```

(4) 我们将为每一个聚类绘制多边形。该函数需要的参数包括Surface对象、颜色（本例中使用红色）和数据点列表。

```python
pygame.draw.polygon(screen, (255, 0, 0), polygon_points[i])
```

绘制结果如下图所示。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214171106495.png" alt="image-20250214171106495" style="zoom:50%;" />

> 注：效果可能和书上不一样，因为是随机生成的。

完整代码如下：

```python
import sklearn
import numpy as np
import pygame,sys
from pygame.locals import *

positions = np.random.randint(0, 400, size=(30, 2))

positions_norms = np.sum(positions ** 2, axis=1)
S = - positions_norms[:, np.newaxis] - positions_norms[np.newaxis, :] + 2 * np.dot(positions, positions.T)

aff_pro = sklearn.cluster.AffinityPropagation().fit(S)
labels = aff_pro.labels_

polygon_points = []
for i in range(max(labels) + 1):
    polygon_points.append([])
for i in range(len(labels)):
    polygon_points[labels[i]].append(positions[i])
pygame.init()
screen = pygame.display.set_mode((400, 400))
while True:
    for i in range(len(polygon_points)):
        pygame.draw.polygon(screen, (255, 0, 0), polygon_points[i])
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()

```

刚才做了些什么 :  下面的表格给出了人工智能示例代码中最重要的几个函数的功能说明。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214171200266.png" alt="image-20250214171200266" style="zoom:50%;" />

# 11.13 OpenGL 和 Pygame

OpenGL是专业的用于二维和三维图形的计算机图形应用程序接口（API），由函数和一些常数构成。我们将重点关注其Python的实现，即`PyOpenGL`。使用如下命令安装PyOpenGL：

```python
pip install  PyOpenGL_accelerate 
```

你可能需要根权限来执行这条命令。以下是相应的easy_install命令：

```python
easy_install PyOpenGL PyOpenGL_accelerate 
```

在pycharm中，一样可以下载这2个依赖：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214171344121.png" alt="image-20250214171344121" style="zoom:50%;" />

备注，如果下载`PyOpenGL_accelerate`失败，参考帖子：https://www.cnblogs.com/sea-stream/p/9840986.html

或者试试conda安装：

```python
conda install PyOpenGL
```

如果用conda安装成功，那么在pycharm中用项目解释器就换成conda的即可。如下图所示：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214173238215.png" alt="image-20250214173238215" style="zoom:50%;" />

# 11.14 动手实践：绘制谢尔宾斯基地毯

为了演示OpenGL的功能，我们将使用OpenGL绘制谢尔宾斯基地毯（Sierpinski gasket），亦称作谢尔宾斯基三角形（Sierpinski triangle）或谢尔宾斯基筛子（Sierpinski sieve）。这是一种三角形形状的分形（fractal），由数学家瓦茨瓦夫·谢尔宾斯基（Waclaw Sierpinski）提出。这个三角形是经过原则上无穷的递归过程得到的。请完成如下步骤绘制谢尔宾斯基地毯。

> 注意书上提供的源代码中，`gluOrtho2D`函数拼错了

(1) 首先，我们将初始化一些OpenGL相关的基本要素，包括设置显示模式和背景颜色等。在本节的末尾可以找到相关函数的详细说明。

```python
def display_openGL(w, h):
    """
    初始化 OpenGL 环境，设置窗口和投影矩阵
    :param w: 窗口的宽度
    :param h: 窗口的高度
    """
    # 设置窗口大小，并启用 OpenGL 渲染和双缓冲
    pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF)

    # 设置背景色为黑色
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 清除颜色缓冲和深度缓冲
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置 2D 正交投影矩阵，使用 gluOrtho2D 函数，设置视口范围
    # 这里将视口范围设定为从 (0, 0) 到 (w, h)，即图像的宽度和高度
    gluOrtho2D(0, w, 0, h)

```

(2) 依据分形的算法，我们应该尽可能多地准确地绘制结点。第一步，我们将绘制颜色设置为红色。第二步，我们定义三角形的顶点。随后，我们定义随机挑选的索引，即从三角形的3个顶点中任意选出其中一个。从三角形靠中间的位置随意指定一点——这个点在哪里并不重要。然后，我们在前一次的点和随机选出的三角形顶点之间的中点处进行绘制。最后，我们强制刷新缓冲以保证绘图命令全部得以执行。

```python
# 设置绘图颜色，这里是红色（RGB）
glColor3f(1.0, 0, 0)

# 定义三角形的三个顶点
# vertices 数组存储的是三角形的 3 个顶点坐标
vertices = np.array([[0, 0], [DIM / 2, DIM], [DIM, 0]])

# 设置绘制的点的数量（分形中的点）
NPOINTS = 9000

# 使用随机生成的整数值来选择 vertices 数组中的一个顶点
indices = np.random.random_integers(0, 2, NPOINTS)

# 初始点的位置
point = [175.0, 150.0]

# 进入循环绘制分形图
for i in range(NPOINTS):
    glBegin(GL_POINTS)  # 开始绘制点
    # 更新当前点的位置：新的点是原点和当前选择的顶点的中点
    point = (point + vertices[indices[i]]) / 2.0
    # 绘制当前点
    glVertex2fv(point)
    glEnd()  # 结束绘制

# 刷新 OpenGL 的渲染缓冲，确保所有绘制的内容立即显示
glFlush()
```

谢尔宾斯基三角形如下图所示。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214173906435.png" alt="image-20250214173906435" style="zoom:50%;" />

完整代码如下：

```python
# 导入所需的库
import pygame
from pygame.locals import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def display_openGL(w, h):
    """
    初始化 OpenGL 环境，设置窗口和投影矩阵
    :param w: 窗口的宽度
    :param h: 窗口的高度
    """
    # 设置窗口大小，并启用 OpenGL 渲染和双缓冲
    pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF)

    # 设置背景色为黑色
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 清除颜色缓冲和深度缓冲
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置 2D 正交投影矩阵，使用 gluOrtho2D 函数，设置视口范围
    # 这里将视口范围设定为从 (0, 0) 到 (w, h)，即图像的宽度和高度
    gluOrtho2D(0, w, 0, h)


def main():
    """
    主函数，初始化 Pygame，设置 OpenGL 渲染，并运行动画
    """
    # 初始化 pygame
    pygame.init()

    # 设置 Pygame 窗口标题
    pygame.display.set_caption('OpenGL Demo')

    # 设置窗口的尺寸
    DIM = 400

    # 调用 display_openGL 函数，设置窗口和 OpenGL 环境
    display_openGL(DIM, DIM)

    # 设置绘图颜色，这里是红色（RGB）
    glColor3f(1.0, 0, 0)

    # 定义三角形的三个顶点
    # vertices 数组存储的是三角形的 3 个顶点坐标
    vertices = np.array([[0, 0], [DIM / 2, DIM], [DIM, 0]])

    # 设置绘制的点的数量（分形中的点）
    NPOINTS = 9000

    # 使用随机生成的整数值来选择 vertices 数组中的一个顶点
    indices = np.random.random_integers(0, 2, NPOINTS)

    # 初始点的位置
    point = [175.0, 150.0]

    # 进入循环绘制分形图
    for i in range(NPOINTS):
        glBegin(GL_POINTS)  # 开始绘制点
        # 更新当前点的位置：新的点是原点和当前选择的顶点的中点
        point = (point + vertices[indices[i]]) / 2.0
        # 绘制当前点
        glVertex2fv(point)
        glEnd()  # 结束绘制

    # 刷新 OpenGL 的渲染缓冲，确保所有绘制的内容立即显示
    glFlush()

    # 更新窗口显示内容
    pygame.display.flip()

    # 事件循环，保持窗口开启直到用户关闭窗口
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                return  # 如果收到关闭事件，则退出主循环


# 如果是直接运行该脚本，调用 main 函数
if __name__ == '__main__':
    main()

```

刚才做了些什么 :如前所述，下面的表格给出了示例代码中最重要的一些函数的功能说明。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214171625902.png" alt="image-20250214171625902" style="zoom:50%;" />

# 11.15 模拟游戏

作为最后一个示例，我们将根据生命游戏（Conway’ s Game of Life）来完成一个模拟生命的游戏。原始的生命游戏是基于几个基本规则的。我们从一个随机初始化的二维方形网格开始。网格中每一个细胞的状态可能是生或死，由其相邻的8个细胞决定。在这个规则下可以使用卷积进行计算，我们需要SciPy的工具包完成卷积运算。

# 11.16 动手实践：模拟生命

下面的代码实现了生命游戏，并做了如下修改：

- 单击鼠标绘制一个十字架；
- 按下r 键将网格重置为随机状态；
- 按下b 键在鼠标位置创建一个方块；
- 按下g 键创建一个形如滑翔机的图案。

本例的代码中最重要的数据结构就是一个二维数组，用于维护游戏界面上像素的颜色值。该数组被随机初始化，然后在游戏主循环中每一轮迭代重新计算一次。在本节的末尾可以找到相关函数的更多信息。

(1) 根据游戏规则，我们将使用卷积进行计算。

```python
def get_pixar(arr, weights):
    """
    获取一个新的 pixar 数组，通过卷积操作计算新状态并根据特定规则返回符合条件的状态。
    :param arr: 当前像素数组
    :param weights: 卷积核，用于对像素进行卷积运算
    :return: 更新后的像素数组
    """
    # 使用 ndimage.convolve 执行卷积运算，'wrap' 模式表示边界循环处理
    states = ndimage.convolve(arr, weights, mode='wrap')

    # 根据状态值判断像素是否符合某些条件
    bools = (states == 13) | (states == 12) | (states == 3)

    # 将布尔数组转换为 0 或 1 的整数数组
    return bools.astype(int)
```

(2) 我们可以使用在第2章中学到的索引技巧绘制十字架。

```python
def draw_cross(pixar):
    """
    在鼠标位置画一个十字形（横向和纵向都设置为 1）
    :param pixar: 当前的像素数组
    """
    (posx, posy) = pygame.mouse.get_pos()  # 获取鼠标当前的屏幕位置
    pixar[posx, :] = 1  # 横向设置为 1
    pixar[:, posy] = 1  # 纵向设置为 1
```

(3) 随机初始化网格：

```python
# 更新的 random_init 函数，使用 np.random.randint 替代 random_integers(以及废除了)
def random_init(n):
    """
    创建一个 n x n 的随机二维数组，每个元素是 0 或 1，用于初始化像素矩阵。
    :param n: 矩阵的大小
    :return: 一个随机的 0 或 1 的二维数组
    """
    return np.random.randint(0, 2, (n, n))  # 生成 0 或 1 的二维数组
```







单机鼠标效果：（绘制一个十字架）

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214180610998.png" alt="image-20250214180610998" style="zoom:50%;" />

按下r 键将网格重置为随机状态：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214180649338.png" alt="image-20250214180649338" style="zoom:50%;" />

按下b 键在鼠标位置创建一个方块:

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214180712833.png" alt="image-20250214180712833" style="zoom:50%;" />

按下g 键创建一个形如滑翔机的图案:(截图效果不是很明显，但是动画可以看得出来)

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214180800406.png" alt="image-20250214180800406" style="zoom:50%;" />

完整代码如下：

```python
import os, pygame
from pygame.locals import *
import numpy as np
from scipy import ndimage


# 更新的 random_init 函数，使用 np.random.randint 替代 random_integers
def random_init(n):
    """
    创建一个 n x n 的随机二维数组，每个元素是 0 或 1，用于初始化像素矩阵。
    :param n: 矩阵的大小
    :return: 一个随机的 0 或 1 的二维数组
    """
    return np.random.randint(0, 2, (n, n))  # 生成 0 或 1 的二维数组


def get_pixar(arr, weights):
    """
    获取一个新的 pixar 数组，通过卷积操作计算新状态并根据特定规则返回符合条件的状态。
    :param arr: 当前像素数组
    :param weights: 卷积核，用于对像素进行卷积运算
    :return: 更新后的像素数组
    """
    # 使用 ndimage.convolve 执行卷积运算，'wrap' 模式表示边界循环处理
    states = ndimage.convolve(arr, weights, mode='wrap')

    # 根据状态值判断像素是否符合某些条件
    bools = (states == 13) | (states == 12) | (states == 3)

    # 将布尔数组转换为 0 或 1 的整数数组
    return bools.astype(int)


def draw_cross(pixar):
    """
    在鼠标位置画一个十字形（横向和纵向都设置为 1）
    :param pixar: 当前的像素数组
    """
    (posx, posy) = pygame.mouse.get_pos()  # 获取鼠标当前的屏幕位置
    pixar[posx, :] = 1  # 横向设置为 1
    pixar[:, posy] = 1  # 纵向设置为 1


def draw_pattern(pixar, pattern):
    """
    根据传入的图案名称在像素矩阵中绘制对应的图案
    :param pixar: 当前的像素数组
    :param pattern: 图案名称（如 glider, block, exploder 等）
    """
    print(pattern)
    # 根据图案选择对应的坐标位置
    if pattern == 'glider':
        coords = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    elif pattern == 'block':
        coords = [(3, 3), (3, 2), (2, 3), (2, 2)]
    elif pattern == 'exploder':
        coords = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 3)]
    elif pattern == 'fpentomino':
        coords = [(2, 3), (3, 2), (4, 2), (3, 3), (3, 4)]

    # 获取鼠标位置作为图案的左上角位置
    pos = pygame.mouse.get_pos()

    # 定义 x 和 y 方向上的间隔
    xs = np.arange(0, pos[0], 10)
    ys = np.arange(0, pos[1], 10)

    # 在这些位置上绘制图案
    for x in xs:
        for y in ys:
            for i, j in coords:
                pixar[x + i, y + j] = 1  # 设置图案坐标为 1，表示活跃状态


def main():
    """
    主函数，初始化 Pygame 和像素数组，并启动事件循环
    """
    # 初始化 Pygame
    pygame.init()

    # 定义矩阵大小
    N = 400

    # 设置窗口的大小
    pygame.display.set_mode((N, N))
    pygame.display.set_caption("Life Demo")  # 设置窗口标题

    # 获取 Pygame 屏幕对象
    screen = pygame.display.get_surface()

    # 初始化像素数组
    pixar = random_init(N)

    # 设置卷积核，用于计算像素的状态变化
    weights = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    cross_on = False  # 控制是否绘制十字形

    while True:
        # 根据卷积核更新像素状态
        pixar = get_pixar(pixar, weights)

        # 如果 cross_on 为 True，绘制十字形
        if cross_on:
            draw_cross(pixar)

        # 使用 Pygame 显示像素数组（乘以 255 ** 3 来调节亮度）
        pygame.surfarray.blit_array(screen, pixar * 255 ** 3)

        # 更新显示
        pygame.display.flip()

        # 处理事件
        for event in pygame.event.get():
            if event.type == QUIT:
                return  # 如果点击关闭按钮，退出程序
            if event.type == MOUSEBUTTONDOWN:
                cross_on = not cross_on  # 切换十字形的绘制状态
            if event.type == KEYDOWN:
                if event.key == ord('r'):  # 按下 'r' 键，随机初始化像素
                    pixar = random_init(N)
                    print("Random init")
                if event.key == ord('g'):  # 按下 'g' 键，绘制 glider 图案
                    draw_pattern(pixar, 'glider')
                if event.key == ord('b'):  # 按下 'b' 键，绘制 block 图案
                    draw_pattern(pixar, 'block')
                if event.key == ord('e'):  # 按下 'e' 键，绘制 exploder 图案
                    draw_pattern(pixar, 'exploder')
                if event.key == ord('f'):  # 按下 'f' 键，绘制 fpentomino 图案
                    draw_pattern(pixar, 'fpentomino')


# 如果是直接运行该脚本，调用 main 函数
if __name__ == '__main__':
    main()

```

刚才做了些什么 :  我们使用的一些NumPy和SciPy的函数需要进一步说明，参见下面的表格。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214181642892.png" alt="image-20250214181642892" style="zoom:50%;" />

# 11.17 本章小结

一开始，你可能会觉得在本书中提到Pygame有些奇怪。希望你在阅读完本章内容后，觉察到一起使用NumPy和Pygame的妙处。毕竟游戏需要很多计算，因此NumPy和SciPy是理想的选择。

游戏也需要人工智能，如`scikit-learn`中可以找到相应的支持。总之，编写游戏是一件有趣的事情，我们希望最后一章的内容如同前面十章教程的正餐之后的甜点或咖啡。如果你还没有“吃饱”，请参阅本书作者的另一本著作《NumPy攻略：Python科学计算与数据分析》，比本书更为深入且与本书内容互不重叠。

本书阅读全部完成。～～～撒花🎉

