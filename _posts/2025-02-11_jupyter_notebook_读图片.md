---
layout: post
title: "读图片-jupyter notebook"
subtitle: "三种方式+图像简单操作"
date: 2025-02-11
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 人工智能AI基础
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>

已有图片`cat.jpg`

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211143332537.png" alt="image-20250211143332537" style="zoom:50%;" />

> 相对于代码的位置，可以用`./cat.jpg`进行读取。

下面是3种读图片的方法。

# 1.python读图片-pillow

图片文件不适合用open去读取

> 用open读图片，易引发`UnicodeDecodeError: 'gbk' codec can't decode byte 0xff in position 0: illegal multibyte sequence`错误。

`PIL` 是 Python Imaging Library 的缩写，它是一个非常流行的图像处理库，提供了广泛的图像处理功能，比如打开、保存、转换、调整大小、裁剪、旋转等操作。

然而，`PIL` 本身已经不再更新和维护，取而代之的是一个称为 **Pillow** 的库。Pillow 是 `PIL` 的一个友好的分支和升级版本，现如今它是使用 `PIL` 功能的标准库。

---

## Pillow读图片

安装依赖（Jupyter notebook）

```python
! pip install Pillow
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211143805172.png" alt="image-20250211143805172" style="zoom:50%;" />

导入依赖读图片：

```python
from PIL import Image
cat = Image.open('./cat.jpg')
cat
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211143920163.png" alt="image-20250211143920163" style="zoom:50%;" />

## 补充：不同格式的图片

JPG 或 JPEG（Joint Photographic Experts Group）是广泛使用的图像压缩格式，主要用于存储彩色照片。

JPEG 图像采用的是**8位颜色深度**的表示方式。（0-255）

每个颜色通道（红色、绿色、蓝色）都使用一个 **8-bit** 数值来表示，即 0 到 255 之间的整数。每个像素由三个颜色通道组成（RGB 模式），每个通道的值都在 0 到 255 之间。这里的 0 代表最暗的颜色，255 代表最亮的颜色。

- **例如**，一个红色像素的 RGB 值可能是 `(255, 0, 0)`，表示完全的红色，绿色和蓝色通道的值是 0。

JPG 图像通常使用压缩算法对数据进行有损压缩，因此它的文件体积较小，但也会牺牲一些图像的细节和质量。

---

PNG（Portable Network Graphics）是一种无损压缩格式，常用于需要保持透明度的图像（例如网页上的图标、图像等）。PNG 支持不同的颜色深度，可以是 **8位** 或 **16位**，但通常以 **8位** 来存储 RGB 图像数据。并且，PNG 格式的图像可以包含 **透明通道**（即 RGBA 模式，其中 A 表示透明度）。

- **像素值范围**：PNG 图像通常使用浮点数表示像素的颜色值，范围从 **0 到 1**。例如，每个颜色通道（R、G、B）的像素值被表示为 0 到 1 之间的小数值，这种方式是浮动值而非整数值。
    - **例如**，一个红色像素的 RGB 值可能是 `(1.0, 0.0, 0.0)`，表示完全的红色，绿色和蓝色通道的值是 0。这些值是浮动的，而不是整数形式。

PNG 是一种 **无损压缩格式**，即不会丢失任何图像细节。虽然它可能比 JPG 图像更大，但保留了原始图像的所有信息。

---

总结：

**JPG/JPEG**：使用 **0 到 255** 范围的整数来表示图像中的每个像素的 RGB 值，适合压缩图像（有损压缩）。

**PNG**：使用 **0 到 1** 之间的小数值来表示图像中的每个像素的 RGB 值，适合需要高质量和无损压缩的图像，特别是在透明通道的处理中。

---

## 对图片进行简单操作

```python
type(cat)  # PIL.JpegImagePlugin.JpegImageFile

cat.size # (730, 456)
```

> 补充：
>
> **`cat.size`**: 这是 `Pillow` 中 `Image` 对象的一个属性，返回图像的尺寸信息。具体来说，`cat.size` 返回一个元组 `(width, height)`，其中：
>
> - `width` 是图像的宽度（以像素为单位）
> - `height` 是图像的高度（以像素为单位）

```python
cat.mode # 'RGB'
```

> 补充：
>
> 除了 RGB，`Pillow` 还支持多种颜色模式，每种模式有不同的表示方式和用途。例如：
>
> - **'L'**：灰度模式（Luminance），表示图像是灰度图像，每个像素只有一个通道，范围是 0 到 255。
> - **'RGBA'**：RGBA 模式表示图像包含红、绿、蓝和透明度（Alpha）通道。每个像素由四个通道的值组成，透明度通道用于表示图像的透明部分。
> - **'CMYK'**：印刷领域使用的颜色模式，表示青色（Cyan）、品红（Magenta）、黄色（Yellow）和黑色（Key）。
> - **'1'**：黑白模式，每个像素只有两个值：0（黑色）和 1（白色）。

```python
cat.getchannel(2)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211144700659.png" alt="image-20250211144700659" style="zoom:50%;" />

`cat.getchannel(2)` 是用于从图像中提取指定通道的一个方法。这里的 `2` 表示我们要获取图像中的第三个颜色通道。

- **0**：红色通道 (R)
- **1**：绿色通道 (G)
- **2**：蓝色通道 (B)

如果图像是 RGBA 模式（即包含透明通道的图像），通道索引 `0`、`1`、`2` 和 `3` 分别代表红色、绿色、蓝色和透明度（Alpha）通道。

---

**可以直接通过np.array把pillow读取的image对象转换成ndarray**

```python
import numpy as np
catArr = np.array(cat)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211144928116.png" alt="image-20250211144928116" style="zoom:50%;" />

注意：

```python
cat.size
catArr.shape
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211145021889.png" alt="image-20250211145021889" style="zoom:50%;" />

**`cat.size`**（来自 `PIL.Image` 对象），图像的尺寸（宽度和高度），表示图像的大小（以像素为单位）。

**`catArr.shape`**（来自 `numpy` 数组），数组的形状（即数组的维度信息）。在将 `PIL.Image` 对象转换为 `numpy` 数组后，图像数据就被存储为一个 **多维数组**。

**456**：图像的 **高度**（行数）

**730**：图像的 **宽度**（列数）

**3**：表示图像是 **RGB** 格式，每个像素包含 3 个颜色通道（红色、绿色、蓝色），因此是一个 3 通道的彩色图像。

即：

**`catArr.shape`** 返回的是 `numpy` 数组的形状，它包含三个维度：

- 第一个维度：图像的高度（即行数）
- 第二个维度：图像的宽度（即列数）
- 第三个维度：图像的颜色通道数（对于 RGB 图像是 3）

----

# 2.opencv读图片

下载依赖（如果有必要，重启内核），下面指定了下载源---豆瓣

> 注：
>
> 豆瓣源: https://pypi.douban.com/simple
>
> 清华源: https://pypi.tuna.tsinghua.edu.cn/simple

```python
!pip install opencv-python -i https://pypi.douban.com/simple
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211145453145.png" alt="image-20250211145453145" style="zoom:50%;" />

导入依赖读图片：

```python
import cv2
cat_cv = cv2.imread('./cat.jpg')
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211145611010.png" alt="image-20250211145611010" style="zoom:50%;" />

opencv默认的颜色空间是BGR

展示图片：

```python
cv2.imshow('cat', cat_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

会有弹窗：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211145710266.png" alt="image-20250211145710266" style="zoom:50%;" />

注：

1.`cat_cv` 是一个 NumPy 数组

2.**`cv2.imshow()`** 用来在一个窗口中显示图像。它的第一个参数是窗口的名称（在这里是 `'cat'`），第二个参数是要显示的图像数据（`cat_cv`）。

3.**`cv2.waitKey(0)`** 会暂停程序的执行，等待用户在显示图像窗口中按下任意键。如果传入的参数是 `0`，表示无限期等待直到用户按下一个键。如果传入的是正整数参数，表示等待指定的毫秒数。如果在这段时间内用户按下键，则继续执行程序。通常用于处理图像时添加时间延迟。

4.**`cv2.destroyAllWindows()`** 会关闭所有由 `cv2.imshow()` 打开的窗口。在图像显示完并等待按键后，调用此函数来销毁所有显示的 OpenCV 窗口。

# 3.`matplotlib`读取图片

导入依赖：

```python
import matplotlib.pyplot as plt
```

读图片：

```python
cat_plt = plt.imread('./cat.jpg')
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211150530460.png" alt="image-20250211150530460" style="zoom:50%;" />

可以看出，也是RGB模式。

展示图片：

```python
plt.imshow(cat_plt)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211150635443.png" alt="image-20250211150635443" style="zoom:50%;" />

输出 `<matplotlib.image.AxesImage at 0x1224862b0>`，这表示你在图像显示过程中实际上得到了一个 **`AxesImage` 对象**。它是 `matplotlib` 用来表示图像数据的对象类型。

**`<matplotlib.image.AxesImage at 0x1224862b0>`**：这是 Python 解释器打印出来的对象的字符串表示。它显示了 `AxesImage` 对象的类型和内存地址（这里是 `0x1224862b0`）。每次你创建新的图像对象时，内存地址会不同，因此这里的地址可能会发生变化。

如果不希望显示，则加一句`plt.axis('off')`即可。

---

## 对图片进行翻转

如何上下翻转, 左右翻转, 上下左右都翻转？颜色翻转？

实质就是操作NumPy数组

> 注意维度信息：
>
> ```python
> cat_plt.shape   # (456, 730, 3)   456高，730宽，3颜色
> ```

```python
# 上下
plt.imshow(cat_plt[::-1])
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211151912082.png" alt="image-20250211151912082" style="zoom:50%;" />

```python
# 左右
plt.imshow(cat_plt[::,::-1])
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211151927884.png" alt="image-20250211151927884" style="zoom:50%;" />

```python
# 上下左右都翻转
plt.imshow(cat_plt[::-1,::-1])
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211151943630.png" alt="image-20250211151943630" style="zoom:50%;" />

```python
# 颜色翻转---》R和B两个转换下
plt.imshow(cat_plt[::,::,::-1])
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250211152005377.png" alt="image-20250211152005377" style="zoom:50%;" />



