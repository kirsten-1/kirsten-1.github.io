---
layout: post
title: "numpy(9)使用Matplotlib绘图"
subtitle: "第 9 章 使用Matplotlib绘图"
date: 2025-02-12
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 人工智能AI基础,NumPy,2025
---


<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>


前8章以及其他补充已经整理如下：

[第1章NumPy入门](https://kirsten-1.github.io/2025/01/27/numpy(1)_%E5%85%A5%E9%97%A8/)

[第2章NumPy基础](https://kirsten-1.github.io/2025/02/04/numpy(2)_numpy%E5%9F%BA%E7%A1%80/)

[第3章常用函数](https://kirsten-1.github.io/2025/02/06/Numpy(3)%E5%B8%B8%E7%94%A8%E5%87%BD%E6%95%B0/)

[广播与广播机制](https://kirsten-1.github.io/2025/02/04/Numpy%E7%9A%84%E5%B9%BF%E6%92%AD%E5%92%8C%E5%B9%BF%E6%92%AD%E6%9C%BA%E5%88%B6/)

[第4章便捷函数](https://kirsten-1.github.io/2025/02/07/NumPy(4)%E4%BE%BF%E6%8D%B7%E5%87%BD%E6%95%B0/)

[第5章矩阵和通用函数](https://kirsten-1.github.io/2025/02/09/NumPy(5)%E7%9F%A9%E9%98%B5%E5%92%8C%E9%80%9A%E7%94%A8%E5%87%BD%E6%95%B0/)

[第6章深入学习NumPy模块](https://kirsten-1.github.io/2025/02/09/NumPy(6)%E6%B7%B1%E5%85%A5%E5%AD%A6%E4%B9%A0NumPy%E6%A8%A1%E5%9D%97/)

[第7章专用函数](https://kirsten-1.github.io/2025/02/10/NumPy(7)%E4%B8%93%E7%94%A8%E5%87%BD%E6%95%B0/)

[第8章质量控制](https://kirsten-1.github.io/2025/02/12/NumPy(8)%E8%B4%A8%E9%87%8F%E6%8E%A7%E5%88%B6/)

----

Matplotlib是一个非常有用的Python绘图库。它和NumPy结合得很好，但本身是一个单独的开源项目。

你可以访问http://matplotlib.sourceforge.net/gallery.html查看美妙的示例图库。

> 这个网站已经不能访问了（这本书比较早了）

Matplotlib中有一些功能函数可以从雅虎财经频道下载并处理数据。我们将看到几个股价图的例子。

本章涵盖以下内容：

- 简单绘图；
-  子图；
- 直方图；
- 定制绘图；
- 三维绘图；
- 等高线图；
- 动画；
- 对数坐标图。

----

# 9.1 简单绘图

`matplotlib.pyplot`包中包含了简单绘图功能。

需要记住的是，随后调用的函数都会改变当前的绘图。最终，我们会将绘图存入文件或使用show函数显示出来。不过如果我们用的是运行在Qt或Wx后端的IPython，图形将会交互式地更新，而不需要等待show函数的结果。这类似于屏幕上输出文本的方式，可以源源不断地打印出来。

---

# 9.2 动手实践：绘制多项式函数

为了说明绘图的原理，我们来绘制多项式函数的图像。我们将使用NumPy的多项式函数`poly1d`来创建多项式。

(1) 以自然数序列作为多项式的系数，使用`poly1d`函数创建多项式。

```python
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float)) 
```

`np.poly1d([1, 2, 3, 4])` 代表：$$1 x^3+2x^2+3x+4$$，`astype(float)` **的作用是将数组中的元素转换为 `float` 类型**，因为`np.array([1, 2, 3, 4])`默认创建的是int类型

(2) 使用NumPy的linspace函数创建x轴的数值，在-10和10之间产生30个均匀分布的值。

```python
x = np.linspace(-10, 10, 30)
```

(3) 计算我们在第一步中创建的多项式的值。

```python
y = func(x)
```

(4) 调用plot函数，这并不会立刻显示函数图像。

```python
plt.plot(x, y)
```

> 注：这一步需要导入：`import matplotlib.pyplot as plt`

(5) 使用xlabel函数添加x轴标签。

```python
plt.xlabel("x")
```

(6) 使用ylabel函数添加y轴标签。

```python
plt.ylabel("y(x)")
```

(7) 调用show函数显示函数图像。

```python
plt.show()
```

绘制的多项式函数如下图所示。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212171316818.png" alt="image-20250212171316818" style="zoom:50%;" />

刚才做了些什么 :我们绘制了多项式函数的图像并显示在屏幕上。我们对x轴和y轴添加了文本标签。

完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.show()
```

## 突击测验：plot函数

问题1 plot函数的作用是什么？ (1) 在屏幕上显示二维绘图的结果 (2) 将二维绘图的结果存入文件 (3) 1和2都是 (4) 1、2、3都不是

> 答案：（4）1、2、3都不是
>
> 因为 `plot` 函数 **本身** 既不会 **自动显示绘图**，也不会 **自动存储绘图**。它的作用只是 **创建一个二维曲线图**，但不负责显示或保存。

# 9.3 格式字符串

plot函数可以接受任意个数的参数。在前面一节中，我们给了两个参数。我们还可以使用可选的格式字符串参数指定线条的颜色和风格，默认为b-即蓝色实线。你可以指定为其他颜色和风格，如红色虚线。

# 9.4 动手实践：绘制多项式函数及其导函数

我们来绘制一个多项式函数，以及使用`deriv`函数和参数m为1得到的其一阶导函数。

> 注意：不是derive，是deriv

我们已经在之前的“动手实践”教程中完成了第一部分。我们希望用两种不同风格的曲线来区分两条函数曲线。

(1) 创建多项式函数及其导函数。

```python
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)
func1 = func.derive(m = 1)
y1 = func1(x)
```

(2)以两种不同风格绘制多项式函数及其导函数：红色圆形和绿色虚线。你可能无法在本书的印刷版中看到彩色图像，因此只能自行尝试绘制图像。

```python
plt.plot(x, y, "ro", x, y1, "g--")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

注：

`'r'` → **红色** (`red`)

`'o'` → **圆形标记** (`circle marker`)

`'g'` → **绿色** (`green`)

`'--'` → **虚线** (`dashed line`)

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212172222623.png" alt="image-20250212172222623" style="zoom:50%;" />

刚才做了些什么 ：我们使用两种不同风格的曲线绘制了一个多项式函数及其导函数，并只调用了一次plot函数。

完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)
func1 = func.deriv(m = 1)
y1 = func1(x)
plt.plot(x, y, "ro", x, y1, "g--")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

# 9.5 子图

绘图时可能会遇到图中有太多曲线的情况，而你希望分组绘制它们。这可以使用subplot函数完成。

# 9.6 动手实践：绘制多项式函数及其导函数

我们来绘制一个多项式函数及其一阶和二阶导函数。为了使绘图更加清晰，我们将绘制3张子图。

(1) 创建多项式函数及其导函数。

```python
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func1 = func.deriv(m = 1)
func2 = func.deriv(m = 2)
x = np.linspace(-10, 10, 30)
y = func(x)
y1 = func1(x)
y2 = func2(x)
```

(2) 使用subplot函数创建第一个子图。该函数的第一个参数是子图的行数，第二个参数是子图的列数，第三个参数是一个从1开始的序号。另一种方式是将这3个参数结合成一个数字，如311。这样，子图将被组织成3行1列。设置子图的标题为Polynomial，使用红色实线绘制。

```python
plt.subplot(311)
plt.plot(x, y, "r-")
plt.title("polynomial")
```

(3) 使用subplot函数创建第二个子图。设置子图的标题为First Derivative，使用蓝色三角形绘制。

```python
plt.subplot(312)
plt.plot(x, y1, "b^")
plt.title("first derivative")
```

(4) 使用subplot函数创建第三个子图。设置子图的标题为Second Derivative，使用绿色圆形绘制。

```python
plt.subplot(313)
plt.plot(x, y2, "go")
plt.title("second derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212173212446.png" alt="image-20250212173212446" style="zoom:50%;" />

> 说句真心话，很丑啊

刚才做了些什么 ：我们使用3种不同风格的曲线在3张子图中分别绘制了一个多项式函数及其一阶和二阶导函数，子图排列成3行1列。

完整代码：

```python
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func1 = func.deriv(m = 1)
func2 = func.deriv(m = 2)
x = np.linspace(-10, 10, 30)
y = func(x)
y1 = func1(x)
y2 = func2(x)
plt.subplot(311)
plt.plot(x, y, "r-")
plt.title("polynomial")
plt.subplot(312)
plt.plot(x, y1, "b^")
plt.title("first derivative")
plt.subplot(313)
plt.plot(x, y2, "go")
plt.title("second derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

# 9.7 财经

Matplotlib可以帮助我们监控股票投资。使用`matplotlib.finance`包中的函数可以从雅虎财经频道（http://finance.yahoo.com/）下载股价数据，并绘制成K线图（candlestick）。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212173403728.png" alt="image-20250212173403728" style="zoom:50%;" />



# 9.8 动手实践：绘制全年股票价格

我们可以使用`matplotlib.finance`包绘制全年的股票价格。获取数据源需要连接到雅虎财经频道。

(1) 将当前的日期减去1年作为起始日期。

> 注：原书用的是`matplotlib.finance`包，但是已经被弃用和废除。因此 **`quotes_historical_yahoo` 和 `candlestick`** 不再可用。
>
> 这个模块以前用于处理和绘制金融数据，但是在新的 `matplotlib` 版本中已不再支持。
>
> 在旧版本的 `matplotlib` 中，`matplotlib.finance` 提供了金融图表的功能，如 **K线图**（Candlestick chart）等。
>
> 但是，从 **matplotlib 3.0** 版本开始，`finance` 模块被移除。
>
> 【解决办法】：使用 `mplfinance` 库
>
> `mplfinance` 是一个独立的库，专门用于绘制金融图表（如K线图）

安装`mplfinance`:

```python
import sys
!{sys.executable} -m pip install mplfinance
```

安装依赖：`yfinance`

```python
import sys
!{sys.executable} -m pip install yfinance
```



这个例子所需依赖如下：

```python
import sys
from datetime import date
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator
```

```python
today = date.today()  # datetime.date(2025, 2, 12)
start = (today.year - 1, today.month, today.day)  # (2024, 2, 12)
```

(2) 我们需要创建所谓的定位器（locator），这些来自matplotlib.dates包中的对象可以在x轴上定位月份和日期。

```python
alldays = DayLocator()
months = MonthLocator()
```

(3) 创建一个日期格式化器（date formatter）以格式化x轴上的日期。该格式化器将创建一个字符串，包含简写的月份和年份。

```python
month_formatter = DateFormatter("%b %Y")
```

(4) 从雅虎财经频道下载股价数据。

```python
# 设置股票符号
symbol = 'DISH'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
```

运行会发现报错：

```python
[*********************100%***********************]  1 of 1 completed

1 Failed download:
['DISH']: YFTzMissingError('$%ticker%: possibly delisted; no timezone found')
```

可能已经 **退市** 或 **无法找到**

可以尝试其他股票

```python
# 使用一个有效的股票代码（例如苹果公司）
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2021-01-01')
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212181854892.png" alt="image-20250212181854892" style="zoom:50%;" />

下载成功。

(5) 创建一个Matplotlib的figure对象——这是绘图组件的顶层容器。

```python
fig = plt.figure()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212190612340.png" alt="image-20250212190612340" style="zoom:50%;" />

(6) 增加一个子图。

```python
ax = fig.add_subplot(111)
```

(7) 将x轴上的主定位器设置为月定位器。该定位器负责x轴上较粗的刻度。

```python
ax.xaxis.set_major_locator(months)
```

(8) 将x轴上的次定位器设置为日定位器。该定位器负责x轴上较细的刻度。

```python
ax.xaxis.set_minor_locator(alldays)
```

(9) 将x轴上的主格式化器设置为月格式化器。该格式化器负责x轴上较粗刻度的标签。

```python
ax.xaxis.set_major_formatter(month_formatter)
```

(10) `matplotlib.finance`(现在换成了`mpf`)包中的一个函数可以绘制K线图。这样，我们就可以使用获取的股价数据来绘制K线图。我们可以指定K线图的矩形宽度，现在先使用默认值。

```python
mpf.plot(data, type='candle', ax=ax, style='charles', volume=True)
```

但是出现报错：`ValueError: Data for column "Open" must be ALL float or int.`

查看数据类型：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212194133132.png" alt="image-20250212194133132" style="zoom:50%;" />

查看数据：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212194619038.png" alt="image-20250212194619038" style="zoom:50%;" />

可以看到：每一列的名字由两部分组成，例如 `Price`, `Close`, `Ticker`, `AAPL` 等。这可能是由于 `yfinance` 返回的多层列索引（`MultiIndex`）导致的。

在这个表格中，`Price`，`Close`，`High` 等作为主列，而 `AAPL` 则是与每个主列相关的附加标签。

检查data的数据结构：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212194753304.png" alt="image-20250212194753304" style="zoom:50%;" />



需要对数据进行处理：

```python
data.columns = data.columns.get_level_values(0)  # 只取第一级作为列名
print("清理后的列名：")
print(data.columns)

# 确保所有列的数据类型是 float 或 int，并清理数据
data = data.astype(float)  # 转换为 float 类型
data = data.dropna()  # 删除包含 NaN 的行

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212194845236.png" alt="image-20250212194845236" style="zoom:50%;" />

绘制K线图：

```python
mpf.plot(data, type='candle', ax=ax, style='charles', volume=True)
```

出现报错：报错信息 `ValueError: 'volume' must be of type 'matplotlib.axis.Axes'` 表示 `mplfinance` 中的 `volume` 参数要求传递的是 `matplotlib` 的 `Axes` 对象，而不是 `True`。

在 `mplfinance` 中，`volume=True` 是用于显示成交量的图层，但在一些版本的 `mplfinance` 中，它要求你为 `volume` 参数传递一个 `Axes` 对象（这是 `matplotlib` 中绘制图表的基础元素）。

可以直接去掉`volume=True `试试：

```python
mpf.plot(data, type='candle', ax=ax, style='charles', volume=True)
```

(11) 将x轴上的标签格式化为日期。为了更好地适应x轴的长度，标签将被旋转。

```python
# 自动旋转日期标签
fig.autofmt_xdate()

# 显示图形
plt.show()

```

最终效果如下：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212195439563.png" alt="image-20250212195439563" style="zoom:50%;" />





完整代码如下：

```python
import sys
from datetime import date
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")

data.columns = data.columns.get_level_values(0)  # 只取第一级作为列名
print("清理后的列名：")
print(data.columns)

# 确保所有列的数据类型是 float 或 int，并清理数据
data = data.astype(float)  # 转换为 float 类型
data = data.dropna()  # 删除包含 NaN 的行


# 设置绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 画K线图
mpf.plot(data, type='candle', ax=ax, style='charles')

# 格式化x轴
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(month_formatter)

# 自动旋转日期标签
fig.autofmt_xdate()

# 显示图形
plt.show()

```

刚才做了些什么 :  我们从雅虎财经频道下载了某股票的全年股价数据，并据此绘制了K线图。

# 9.9 直方图

直方图（histogram）可以将数据的分布可视化。Matplotlib中有便捷的hist函数可以绘制直方图。该函数的参数中有这样两项——包含数据的数组以及柱形的数量。

# 9.10 动手实践：绘制股价分布直方图

我们来绘制从雅虎财经频道下载的股价数据的分布直方图。

(1) 下载一年以来的数据：

```python
# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
data
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212195650280.png" alt="image-20250212195650280" style="zoom:50%;" />

(2) 上一步得到的股价数据存储在Python列表中。将其转化为NumPy数组并提取出收盘价 数据：

注：

- **使用 `yf.download` 获取的数据** 本质上是一个 `pandas DataFrame`。
- 使用 `.values` 或 `.to_numpy()` 方法将 `DataFrame` 转换为 NumPy 数组。

```python
# 将整个 DataFrame 转换为 NumPy 数组
data_numpy = data.to_numpy()

# 提取收盘价数据（第 0 列，索引为 0）
close_prices = data_numpy[:, 0]  # 取所有行的第 3 列（即 Close）

# 打印收盘价（Close）数据
print(close_prices)
```

(3) 指定合理数量的柱形，绘制分布直方图：

```python
plt.hist(close_prices, int(np.sqrt(len(close_prices))))  
plt.show() 
```

按照书上的代码会出现报错：`TypeError: `bins` must be an integer, a string, or an array`

所以我改为了上面的代码。(数据转换了一下而已)

直方图结果如下：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212200312980.png" alt="image-20250212200312980" style="zoom:50%;" />

完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import mplfinance as mpf
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# 将整个 DataFrame 转换为 NumPy 数组
data_numpy = data.to_numpy()

# 提取收盘价数据（第 0 列，索引为 0）
close_prices = data_numpy[:, 0]  # 取所有行的第 3 列（即 Close）

# 绘制直方图
plt.hist(close_prices, int(np.sqrt(len(close_prices))))  
plt.show() 
```



## 勇敢出发：绘制钟形曲线

使用股价的平均值结合标准差绘制一条钟形曲线（即高斯分布或正态分布）。当然，这只是作为练习。

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")

# 将整个 DataFrame 转换为 NumPy 数组
data_numpy = data.to_numpy()

# 提取收盘价数据（第 3 列，索引为 3）
close_prices = data_numpy[:, 3]  # 取所有行的第 3 列（即 Close）

# 计算股价的平均值和标准差
mean_price = np.mean(close_prices)
std_price = np.std(close_prices)

# 绘制直方图
plt.hist(close_prices, bins=int(np.sqrt(len(close_prices))), density=True, alpha=0.6, color='g')

# 生成正态分布的 x 数据
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# 使用 NumPy 计算正态分布的概率密度函数（PDF）
pdf = (1 / (std_price * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_price) / std_price) ** 2)

# 绘制高斯（正态）分布曲线
plt.plot(x, pdf, 'k', linewidth=2)

# 设置标题和标签
title = f"Fit results: Mean = {mean_price:.2f},  Std Dev = {std_price:.2f}"
plt.title(title)
plt.xlabel('Close Price')
plt.ylabel('Density')

# 显示图形
plt.show()

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212200815612.png" alt="image-20250212200815612" style="zoom:50%;" />

# 9.11 对数坐标图

当数据的变化范围很大时，对数坐标图（logarithmic plot）很有用。Matplotlib中有**semilogx函数（对x轴取对数）、semilogy函数（对y轴取对数）和loglog函数（同时对x轴和y轴取对数）**。

# 9.12 动手实践：绘制股票成交量

股票成交量变化很大，因此我们需要对其取对数后再绘制。首先，我们需要从雅虎财经频道下载历史数据，从中提取出日期和成交量数据，创建定位器和日期格式化器，创建图像并以子图的方式添加。在前面的“动手实践”教程中我们已经完成过这些步骤，因此这里不再赘述。

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
display(data)
# 提取日期
dates = data.index.to_numpy()
# 提取成交量
volume = data['Volume'].to_numpy()
```

(1) 使用对数坐标绘制成交量数据。

```python
plt.semilogy(dates, volume) 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212202508666.png" alt="image-20250212202508666" style="zoom:50%;" />

现在，我们将设置定位器并将x轴格式化为日期。你可以在前一节中找到这些步骤的说明。使用对数坐标图绘制的股票成交量如下图所示。

```python
fig = plt.figure() 
ax = fig.add_subplot(111) 
plt.semilogy(dates, volume) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
fig.autofmt_xdate() 
plt.show 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212202828221.png" alt="image-20250212202828221" style="zoom:50%;" />

完整代码：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
display(data)
# 提取日期
dates = data.index.to_numpy()
# 提取成交量
volume = data['Volume'].to_numpy()

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")

fig = plt.figure() 
ax = fig.add_subplot(111) 
plt.semilogy(dates, volume) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
fig.autofmt_xdate() 
plt.show 
```

# 9.13 散点图

散点图（scatter plot）用于绘制同一数据集中的两种数值变量。Matplotlib的**scatter函数可以创建散点图**。我们可以指定数据点的**颜色和大小，以及图像的alpha透明度**。

# 9.14 动手实践：绘制股票收益率和成交量变化的散点图

我们可以便捷地绘制股票收益率和成交量变化的散点图。同样，我们先从雅虎财经频道下载所需的数据。

(1) 得到的数据存储在Python列表中。将其转化为NumPy数组并提取出收盘价和成交量数据。

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
display(data)
# 提取收盘价
close = data['Close'].to_numpy()
# 提取成交量
volume = data['Volume'].to_numpy()

```

(2) 计算股票收益率和成交量的变化值。

> 在这步之前需要注意，得到的`close`和`volume`都是二维的，建议降维。
>
> 首先要明确 `close` 和 `volume` 是 `pandas Series` 类型，且它们原本是列数据。

```python
# 提取收盘价和成交量并确保它们是一维数组
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
volume = data['Volume'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
```

然后计算股票收益率和成交量变化。

```python
ret = np.diff(close)/close[:-1]  
volchange = np.diff(volume)/volume[:-1] 
print("股票收益率：", ret)
print("成交量变化：", volchange)
```

(3) 创建一个Matplotlib的figure对象。

```python
fig = plt.figure() 
```

(4) 在图像中添加一个子图。

```python
fig.add_subplot(111)
```

(5) 创建散点图，并使得数据点的颜色与股票收益率相关联，数据点的大小与成交量的变化相关联。

```python
ax.scatter(ret, volchange, c=ret * 100, s=volchange * 100, alpha=0.5) 
```

(6) 设置图像的标题并添加网格线。

```python

```

发现绘制的散点图是空白的：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212210419604.png" alt="image-20250212210419604" style="zoom:50%;" />

查看股票收益率和成交量变化的极值：

```python
print("股票收益率：", np.min(ret), np.max(ret))
print("成交量变化：", np.min(volchange), np.max(volchange))
```

结果是：

```python
股票收益率： -0.04816697992219063 0.07264912239393996
成交量变化： -0.8300928298270459 3.7719930579368777
```

数据要进行适当的缩放：

```python
# 进行适当的缩放和调整
ret_scaled = ret * 100  # 放大收益率的范围，确保颜色的可视化更清晰
volchange_scaled = np.clip(volchange, -0.5, 1.0) * 200  # 将成交量变化进行裁剪并缩放，避免过大点
# 绘制散点图
scatter = ax.scatter(ret, volchange, c=ret_scaled, s=volchange_scaled, alpha=0.5, cmap='viridis')
```

此时出现报错：

```
RuntimeWarning: invalid value encountered in sqrt
  scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor
```

出现的 `RuntimeWarning: invalid value encountered in sqrt` 错误通常是由于 **`s` 参数**（即散点大小）中包含 **负数** 或 **`NaN`** 值造成的。Matplotlib 在计算点的大小时使用了平方根，这意味着如果数据中有负值，平方根无法计算，从而导致该警告。

解决：

```python
# 进行适当的缩放和调整
ret_scaled = ret * 100  # 放大收益率的范围，确保颜色的可视化更清晰
volchange_scaled = np.abs(volchange)  # 取成交量变化的绝对值，避免负数
volchange_scaled = np.clip(volchange_scaled, 0, 1) * 200  # 将成交量变化进行裁剪并缩放，避免过大点

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212210455321.png" alt="image-20250212210455321" style="zoom:50%;" />

完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取收盘价和成交量并确保它们是一维数组
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
volume = data['Volume'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

ret = np.diff(close)/close[:-1]  
volchange = np.diff(volume)/volume[:-1] 
# 进行适当的缩放和调整
ret_scaled = ret * 100  # 放大收益率的范围，确保颜色的可视化更清晰
volchange_scaled = np.abs(volchange)  # 取成交量变化的绝对值，避免负数
volchange_scaled = np.clip(volchange_scaled, 0, 1) * 200  # 将成交量变化进行裁剪并缩放，避免过大点

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制散点图
scatter = ax.scatter(ret, volchange, c=ret_scaled, s=volchange_scaled, alpha=0.5, cmap='viridis')

# 添加标题和网格
ax.set_title('Close and Volume Returns')
ax.grid(True)

# 添加颜色条
plt.colorbar(scatter, label='Stock Return (%)')

# 显示图形
plt.show()

```

# 9.15 着色

`fill_between`函数使用指定的颜色填充图像中的区域。我们也可以选择alpha通道的取值。该函数的where参数可以指定着色的条件。

# 9.16 动手实践：根据条件进行着色

假设你想对股票曲线图进行着色，并将低于均值和高于均值的收盘价填充为不同颜色。

`fill_between`函数是完成这项工作的最佳选择。我们仍将省略下载一年以来历史数据、提取日期和收盘价数据以及创建定位器和日期格式化器的步骤。

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
display(data)
# 提取日期和收盘价
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")


```

(1) 创建一个Matplotlib的figure对象。

```python
fig = plt.figure()
```

(2) 在图像中添加一个子图。

```python
ax = fig.add_subplot(111)
```

(3) 绘制收盘价数据。

```python
ax.plot(dates, close)
```

此时可以看到结果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212211128717.png" alt="image-20250212211128717" style="zoom:50%;" />

(4) 对收盘价下方的区域进行着色，依据低于或高于平均收盘价使用不同的颜色填充。

```python
plt.fill_between(dates, close.min(), close, where=close>close.mean(), facecolor="green", alpha=0.4)  
plt.fill_between(dates, close.min(), close, where=close<close.mean(), facecolor="red", alpha=0.4) 
```

现在，我们将设置定位器并将x轴格式化为日期，从而完成绘制。根据条件进行着色的股价如下图所示。

```python
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.grid(True) 
fig.autofmt_xdate() 
plt.show() 
```

最终效果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212211450070.png" alt="image-20250212211450070" style="zoom:50%;" />

完整代码：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
display(data)
# 提取日期和收盘价
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dates, close)

print(close.mean())

plt.fill_between(dates, close.min(), close, where=close>close.mean(), facecolor="green", alpha=0.4)  
plt.fill_between(dates, close.min(), close, where=close<close.mean(), facecolor="red", alpha=0.4) 

ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.grid(True) 
fig.autofmt_xdate() 
plt.show() 
```

# 9.17 图例和注释

对于高质量的绘图，图例和注释是至关重要的。我们可以用legend函数创建透明的图例，并由Matplotlib自动确定其摆放位置。同时，我们可以用annotate函数在图像上精确地添加注释，并有很多可选的注释和箭头风格。

# 9.18 动手实践：使用图例和注释

在第3章中我们学习了如何计算股价的指数移动平均线。我们将绘制一只股票的收盘价和对应的三条指数移动平均线。为了清楚地描述图像的含义，我们将添加一个图例，并用注释标明两条平均曲线的交点。部分重复的步骤将被略去。

(1) 计算并绘制指数移动平均线：如果需要，请回到第3章中复习一下指数移动平均线的计算方法。分别使用9、12和15作为周期数计算和绘制指数移动平均线。

```python
emas = [] 
for i in range(9, 18, 3): 
    weights = np.exp(np.linspace(-1., 0., i))  
    weights /= weights.sum() 
    ema = np.convolve(weights, close)[i-1:-i+1]  
    idx = (i - 6)/3 
    ax.plot(dates[i-1:], ema, lw=idx, label="EMA(%s)" % (i))  
    data = np.column_stack((dates[i-1:], ema))  
    emas.append(np.rec.fromrecords( data, names=["dates", "ema"])) 
```

运行以上代码会出现报错：

```python
DTypePromotionError: The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes.Float64DType'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Float64DType'>)
```

错误 `DTypePromotionError` 是由于 `dates[i-1:]` 和 `ema` 数据类型不兼容导致的。具体来说，`dates[i-1:]` 是 `datetime64` 类型，而 `ema` 是 `float64` 类型。`np.column_stack` 无法将这两种不同类型的数据合并为一个数组，除非将它们转换为兼容的类型。

解决办法：可以将 `dates` 转换为字符串类型或将 `ema` 转换为对象类型，这样它们就能一起存储在一个结构化数组中。

```python
# 创建一个空的列表来存储 EMA 数据
emas = []
# 计算不同窗口大小的指数移动平均（EMA）
for i in range(9, 18, 3):
    weights = np.exp(np.linspace(-1., 0., i))  
    weights /= weights.sum()  
    
    # 计算 EMA
    ema = np.convolve(close, weights, mode='valid')  
    idx = (i - 6) / 3  
    
    # 绘制 EMA 曲线
    ax.plot(dates[i-1:len(ema)+i-1], ema, lw=idx, label=f"EMA({i})")

    # **解决 DTypePromotionError**
    dates_str = dates[i-1:len(ema)+i-1].astype(str)  # 将日期转换为字符串
    data = np.column_stack((dates_str, ema))  # 合并字符串日期和浮点数
    emas.append(np.rec.fromrecords(data, names=["dates", "ema"]))  # 存储结构化数组
```

注意，调用plot函数时需要指定图例的标签。我们将指数移动平均线的值存在数组中，为下一步做准备。

(2) 我们来找到两条指数移动平均曲线的交点。

```python
first = emas[0]["ema"].flatten()  
second = emas[1]["ema"].flatten() 
bools = np.abs(first[-len(second):] - second)/second < 0.0001  
xpoints = np.compress(bools, emas[1]) 
```

按照书上的代码（如上），又出现报错：

```python
UFuncTypeError: ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U18'), dtype('<U18')) -> None
```

表明 **`first` 和 `second` 是字符串 (`dtype('<U18')`)，但 `np.abs(first - second)` 需要数值类型**。

解决：在 `np.rec.fromrecords()` 之前转换数据类型

```python
# 计算不同窗口大小的指数移动平均（EMA）
emas = []
for i in range(9, 18, 3):
    weights = np.exp(np.linspace(-1., 0., i))  
    weights /= weights.sum()  
    
    # 计算 EMA
    ema = np.convolve(close, weights, mode='valid')  
    idx = (i - 6) / 3  
    
    # 绘制 EMA 曲线
    ax.plot(dates[i-1:len(ema)+i-1], ema, lw=idx, label=f"EMA({i})")

    # **解决 DTypePromotionError**
    dates_str = dates[i-1:len(ema)+i-1].astype(str)  # 日期转换为字符串
    ema_float = ema.astype(float)  # 确保 ema 是 float
    data = np.column_stack((dates_str, ema_float))  # 合并数据
    emas.append(np.rec.fromrecords(data, names=["dates", "ema"]))  # 存储结构化数组

# 确保数据类型为 float
first = emas[0]["ema"].astype(float).flatten()  
second = emas[1]["ema"].astype(float).flatten()

# 计算误差并筛选数据
bools = np.abs(first[-len(second):] - second) / second < 0.0001  
xpoints = np.compress(bools, emas[1])  # 过滤数据
```

(3) 我们将找到的交点用注释和箭头标注出来，并确保注释文本在交点的不远处。

```python
for xpoint in xpoints: 
    ax.annotate('x', xy=xpoint, textcoords='offset points',                   
                xytext=(-50, 30), 
                arrowprops=dict(arrowstyle="->")) 
```

(4) 添加一个图例并由Matplotlib自动确定其摆放位置。

```python
leg = ax.legend(loc='best', fancybox=True) 
```

(5) 设置alpha通道值，将图例透明化。

```python
leg.get_frame().set_alpha(0.5) 
```

此时完整代码如下：(但是出现报错)

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")


fig = plt.figure() 
ax = fig.add_subplot(111) 

# 计算不同窗口大小的指数移动平均（EMA）
emas = []
for i in range(9, 18, 3):
    weights = np.exp(np.linspace(-1., 0., i))  
    weights /= weights.sum()  
    
    # 计算 EMA
    ema = np.convolve(close, weights, mode='valid')  
    idx = (i - 6) / 3  
    
    # 绘制 EMA 曲线
    ax.plot(dates[i-1:len(ema)+i-1], ema, lw=idx, label=f"EMA({i})")

    # **解决 DTypePromotionError**
    dates_str = dates[i-1:len(ema)+i-1].astype(str)  # 日期转换为字符串
    ema_float = ema.astype(float)  # 确保 ema 是 float
    data = np.column_stack((dates_str, ema_float))  # 合并数据
    emas.append(np.rec.fromrecords(data, names=["dates", "ema"]))  # 存储结构化数组

# 确保数据类型为 float
first = emas[0]["ema"].astype(float).flatten()  
second = emas[1]["ema"].astype(float).flatten()

# 计算误差并筛选数据
bools = np.abs(first[-len(second):] - second) / second < 0.0001  
xpoints = np.compress(bools, emas[1])  # 过滤数据

for xpoint in xpoints: 
    ax.annotate('x', xy=xpoint, textcoords='offset points',                   
                xytext=(-50, 30), 
                arrowprops=dict(arrowstyle="->")) 

leg = ax.legend(loc='best', fancybox=True) 
leg.get_frame().set_alpha(0.5) 

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y") 
ax.plot(dates, close, lw=1.0, label="Close") 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.grid(True) 
fig.autofmt_xdate() 
plt.show() 
```

报错信息：

```python
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

以及：

```python
ConversionError: Failed to convert value(s) to axis units: '2024-09-20T00:00:00.000000000'
```

这些错误表明 `dates`（时间轴数据）和 `xpoints`（被 `np.compress()` 过滤后用于 `ax.annotate()` 的数据）之间的 **数据类型不匹配**。

修改完，完整代码：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")


fig = plt.figure() 
ax = fig.add_subplot(111) 

# 计算不同窗口大小的指数移动平均（EMA）
emas = []
for i in range(9, 18, 3):
    weights = np.exp(np.linspace(-1., 0., i))  
    weights /= weights.sum()  
    
    # 计算 EMA
    ema = np.convolve(close, weights, mode='valid')  
    idx = (i - 6) / 3  
    
    # 绘制 EMA 曲线
    ax.plot(dates[i-1:len(ema)+i-1], ema, lw=idx, label=f"EMA({i})")

    # **解决 DTypePromotionError**
    dates_str = dates[i-1:len(ema)+i-1].astype(str)  # 日期转换为字符串
    ema_float = ema.astype(float)  # 确保 ema 是 float
    data = np.column_stack((dates_str, ema_float))  # 合并数据
    emas.append(np.rec.fromrecords(data, names=["dates", "ema"]))  # 存储结构化数组

# 确保数据类型为 float
first = emas[0]["ema"].astype(float).flatten()  
second = emas[1]["ema"].astype(float).flatten()

# 计算误差并筛选数据
bools = np.abs(first[-len(second):] - second) / second < 0.0001  

# 检查 `xpoints` 是否为空，避免索引错误
if xpoints.size == 0:
    print("⚠️ 警告: xpoints 为空，没有满足条件的数据点，跳过注释绘制。")
else:
    # 确保 `xpoints` 数据类型正确
    xpoints = np.array([(x['dates'], x['ema']) for x in emas[1] if x['ema'] is not None], 
                        dtype=[('dates', 'datetime64[D]'), ('ema', 'float64')])

    for xpoint in xpoints:
        ax.annotate('x', xy=(xpoint['dates'], xpoint['ema']), textcoords='offset points',                   
                    xytext=(-50, 30), 
                    arrowprops=dict(arrowstyle="->"))


leg = ax.legend(loc='best', fancybox=True) 
leg.get_frame().set_alpha(0.5) 

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y") 
ax.plot(dates, close, lw=1.0, label="Close") 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.grid(True) 
fig.autofmt_xdate() 
plt.show() 
```



包含图例和注释的股价及指数移动平均线图如下所示。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212214003373.png" alt="image-20250212214003373" style="zoom:50%;" />

黑色的这些密集的线条看起来像是 **Matplotlib 的 `ax.annotate()` 注释箭头**，因为它们沿着价格曲线排列，并且看上去像是大量的标注文本或者箭头重叠在一起。

- **`xpoints` 可能包含了过多的点**，导致 `ax.annotate()` 在 **每个数据点** 都画了箭头，导致画面变得混乱。
- **`xytext=(-50, 30)`** 可能导致箭头文本重叠，因为它们都是固定的偏移量，导致文本和箭头覆盖整个图表。
- **布尔条件 `bools` 可能过于宽松**，导致 **过多的数据点** 被选中并进行标注。

简单进行修正：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'AAPL'
if len(sys.argv) == 2:
    symbol = sys.argv[1]

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

# 定义日期定位器和格式化器
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")


fig = plt.figure() 
ax = fig.add_subplot(111) 

# 计算不同窗口大小的指数移动平均（EMA）
emas = []
for i in range(9, 18, 3):
    weights = np.exp(np.linspace(-1., 0., i))  
    weights /= weights.sum()  
    
    # 计算 EMA
    ema = np.convolve(close, weights, mode='valid')  
    idx = (i - 6) / 3  
    
    # 绘制 EMA 曲线
    ax.plot(dates[i-1:len(ema)+i-1], ema, lw=idx, label=f"EMA({i})")

    # **解决 DTypePromotionError**
    dates_str = dates[i-1:len(ema)+i-1].astype(str)  # 日期转换为字符串
    ema_float = ema.astype(float)  # 确保 ema 是 float
    data = np.column_stack((dates_str, ema_float))  # 合并数据
    emas.append(np.rec.fromrecords(data, names=["dates", "ema"]))  # 存储结构化数组

# 确保数据类型为 float
first = emas[0]["ema"].astype(float).flatten()  
second = emas[1]["ema"].astype(float).flatten()

# 计算误差并筛选数据
bools = np.abs(first[-len(second):] - second) / second < 0.0005  


# 检查 `xpoints` 是否为空，避免索引错误
if xpoints.size == 0:
    print("⚠️ 警告: xpoints 为空，没有满足条件的数据点，跳过注释绘制。")
else:
    # 确保 `xpoints` 数据类型正确
    xpoints = np.array([(x['dates'], x['ema']) for x in emas[1] if x['ema'] is not None], 
                        dtype=[('dates', 'datetime64[D]'), ('ema', 'float64')])

    # 减少标注的数量
    xpoints = xpoints[::10]  # 每隔10个点标注一次，避免太多注释

    for idx, xpoint in enumerate(xpoints):
        ax.annotate('x', xy=(xpoint['dates'], xpoint['ema']), textcoords='offset points',
                    xytext=(-30, 30 if idx % 2 == 0 else -30),  # 交错偏移
                    arrowprops=dict(arrowstyle="->"))


leg = ax.legend(loc='best', fancybox=True) 
leg.get_frame().set_alpha(0.5) 

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y") 
ax.plot(dates, close, lw=1.0, label="Close") 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.grid(True) 
fig.autofmt_xdate() 
plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212214406102.png" alt="image-20250212214406102" style="zoom:50%;" />

# 9.19 三维绘图

三维绘图非常壮观华丽，因此我们必须涵盖这部分内容。对于3D作图，我们需要一个和三维投影相关的Axes3D对象。

# 9.20 动手实践：在三维空间中绘图

我们将在三维空间中绘制一个简单的三维函数。$$z = x^2=y^2$$

(1) 我们需要使用3d关键字来指定图像的三维投影。

```python
ax = fig.add_subplot(111, projection='3d') 
```

(2) 我们将使用meshgrid函数创建一个二维的坐标网格。这将用于变量x和y的赋值。

```python
u = np.linspace(-1, 1, 100) 
x, y = np.meshgrid(u, u) 
```

而z是：

```python
z = x ** 2 + y ** 2
```



(3) 我们将指定行和列的步幅，以及绘制曲面所用的色彩表（color map）。步幅决定曲面上“瓦片”的大小，而色彩表的选择取决于个人喜好。

```python
from matplotlib import cm 
ax.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.YlGnBu_r) 
```

3D绘图的结果如下所示。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212214839114.png" alt="image-20250212214839114" style="zoom:50%;" />

完整代码如下：

```python
from matplotlib import cm 
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d') 
u = np.linspace(-1, 1, 100) 
x, y = np.meshgrid(u, u) 
z = x ** 2 + y ** 2
ax.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.YlGnBu_r) 
plt.show()
```

# 9.21 等高线图

Matplotlib中的等高线3D绘图有两种风格——填充的和非填充的。我们可以使用contour函数创建一般的等高线图。对于色彩填充的等高线图，可以使用contourf绘制。

# 9.22 动手实践：绘制色彩填充的等高线图

我们将对前面“动手实践”中的三维数学函数绘制色彩填充的等高线图。代码也非常简单，一个重要的区别是我们不再需要指定三维投影的参数。使用下面这行代码绘制等高线图：

```python
ax.contourf(x, y, z) 
```

完整代码：

```python
fig = plt.figure() 
ax = fig.add_subplot(111) 
u = np.linspace(-1, 1, 100) 
x, y = np.meshgrid(u, u) 
z = x ** 2 + y ** 2
ax.contourf(x, y, z) 
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212221016407.png" alt="image-20250212221016407" style="zoom:50%;" />

如果加上3D参数：

```python
fig = plt.figure() 
# ax = fig.add_subplot(111) 
ax = fig.add_subplot(111, projection='3d') 
u = np.linspace(-1, 1, 100) 
x, y = np.meshgrid(u, u) 
z = x ** 2 + y ** 2
ax.contourf(x, y, z) 
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212221115940.png" alt="image-20250212221115940" style="zoom:50%;" />

# 9.23 动画

Matplotlib提供酷炫的动画功能。Matplotlib中有专门的动画模块。我们需要定义一个回调函数，用于定期更新屏幕上的内容。我们还需要一个函数来生成图中的数据点。

# 9.24 动手实践：制作动画

我们将绘制三个随机生成的数据集，分别用圆形、小圆点和三角形来显示。不过，我们将只用随机值更新其中的两个数据集。

> 注意导入依赖：`import matplotlib.animation as animation `

(1) 我们将用不同颜色的圆形、小圆点和三角形来绘制三个数据集中的数据点。

```python
circles, triangles, dots = ax.plot(x, 'ro', y, 'g^', z, 'b.') 
```

(2) 下面的函数将被定期调用以更新屏幕上的内容。我们将随机更新两个数据集中的y坐标值。

```python
def update(data): 
    circles.set_ydata(data[0])  
    triangles.set_ydata(data[1])  
    return circles, triangles
```

(3) 使用NumPy生成随机数。

```python
def generate(): 
    while True: yield np.random.rand(2, N) 
```

完整代码：（但是会报错）

```python
/var/folders/9l/l51cpj6n21j2pth2z6x42dwc0000gn/T/ipykernel_6215/916763211.py:23: UserWarning: frames=<function generate at 0x1256599d0> which we can infer the length of, did not pass an explicit *save_count* and passed cache_frame_data=True.  To avoid a possibly unbounded cache, frame data caching has been disabled. To suppress this warning either pass `cache_frame_data=False` or `save_count=MAX_FRAMES`.
  anim = animation.FuncAnimation(fig, update, generate, interval=150)
```

在 `FuncAnimation` 中传递了一个生成器函数 (`generate`) 作为 `frames` 参数，但是没有明确指定 `save_count` 参数。这个警告的原因是，生成器没有明确的帧数限制，可能会导致无限制的缓存，这样就可能会导致内存问题。

改进：

```python
anim = animation.FuncAnimation(fig, update, generate, interval=150, cache_frame_data=False)
```

效果：（明明是动画，但是就1帧...）

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212221729224.png" alt="image-20250212221729224" style="zoom:50%;" />

# 9.25 本章小结

本章围绕Matplotlib——一个Python绘图库展开，涵盖简单绘图、直方图、定制绘图、子图、3D绘图、等高线图和对数坐标图等内容。我们还学习了几个绘制股票数据的例子。

显然，我们还只是领略了冰山一角。Matplotlib的功能非常丰富，因此我们没有足够的篇幅来讲述LaTex支持、极坐标支持以及其他功能。

Matplotlib的作者John Hunter于2012年8月离开了我们。

本书的审稿人之一建议在此提及John Hunter纪念基金（John Hunter Memorial Fund，请访问http://numfocus.org/johnhunter/）。

该基金由NumFocus Foundation发起，可以这么说，它给了我们这些John Hunter作品的粉丝们一个回报的机会。更多详情，请访问前面的NumFocus网站链接。

下一章中，我们将学习SciPy——一个建立在NumPy之上的Python科学计算架构。 

