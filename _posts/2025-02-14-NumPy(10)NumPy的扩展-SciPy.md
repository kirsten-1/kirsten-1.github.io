---
layout: post
title: "numpy(10)NumPy的扩展：SciPy"
subtitle: "第 10 章 NumPy 的扩展：SciPy"
date: 2025-02-12
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 人工智能AI基础
---


<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>


前8章以及其他补充已经整理如下：

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

----

SciPy是世界著名的Python开源科学计算库，建立在NumPy之上。它增加的功能包括数值积分、最优化、统计和一些专用函数。很多有一些**高阶抽象和物理模型**需要使用 Scipy。

本章涵盖以下内容：

- 文件输入/输出；
- 统计；
- 信号处理；
- 最优化；
- 插值；
- 图像和音频处理。

---

# 10.1 MATLAB 和 Octave

MATLAB以及其开源替代品Octave都是流行的数学工具。`scipy.io`包的函数可以在Python中加载或保存MATLAB和Octave的矩阵和数组。`loadmat`函数可以加载`.mat`文件。`savemat`函数可以将数组和指定的变量名字典保存为`.mat`文件。

# 10.2 动手实践：保存和加载.mat 文件

如果我们一开始使用了NumPy数组，随后希望在MATLAB或Octave环境中使用这些数组，那么最简单的办法就是创建一个`.mat`文件，然后在MATLAB或Octave中加载这个文件。请完成如下步骤。

(1) 创建NumPy数组并调用`savemat`创建一个`.mat`文件。该函数有两个参数——一个文件名和一个包含变量名和取值的字典。

```python
from scipy import io

a = np.arange(7)
io.savemat("a.mat", {'array': a})
```

(2) 在MATLAB或Octave环境中加载.mat文件，并检查数组中存储的元素。

> 因为我已经卸载matlab，此处以加载这个文件作为检查。

```python
data = io.loadmat("a.mat")
data
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212223326552.png" alt="image-20250212223326552" style="zoom:50%;" />

文件已经正常加载，发现就是存储的ndarray。

刚才做了些什么 ：我们使用NumPy代码创建了一个`.mat文件`并检查了之前创建的NumPy数组的元素。

完整代码：

```python
import numpy as np
from scipy import io

a = np.arange(7)
io.savemat("a.mat", {'array': a})
data = io.loadmat("a.mat")
display(data)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212223505573.png" alt="image-20250212223505573" style="zoom:50%;" />

## 突击测验：加载.mat类型的文件

问题1 以下哪个函数可以加载.mat类型的文件？

(1) Loadmatlab
(2) loadmat
(3) loadoct
(4) frommat

> 答案：（2）loadmat，其余都不是一个有效的 Python 函数

# 10.3 统计

SciPy的统计模块是`scipy.stats`，其中有一个类是连续分布的实现，一个类是离散分布的实现。此外，该模块中还有很多用于统计检验的函数。

# 10.4 动手实践：分析随机数

我们将按正态分布生成随机数，并使用`scipy.stats`包中的统计函数分析生成的数据。请完成如下步骤。

(1) 使用`scipy.stats`包按正态分布生成随机数。

```python
from scipy import stats

generated = stats.norm.rvs(size = 900)
```

(2) 用正态分布去拟合生成的数据，得到其均值和标准差：

```python
stats.norm.fit(generatedrated)
```

均值和标准差如下所示：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212223946959.png" alt="image-20250212223946959" style="zoom:50%;" />

(3) 偏度（skewness）描述的是概率分布的偏斜（非对称）程度。我们来做一个偏度检验。该检验有两个返回值，其中第二个返回值为p-value，即观察到的数据集服从正态分布的概率，取值范围为0~1。

```python
stats.skewtest(generated)
```

偏度检验返回的结果如下：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212224332833.png" alt="image-20250212224332833" style="zoom:50%;" />

因此，该数据集有79%的概率服从正态分布。

> 注：不同的随机数生成，会导致这里不同的`pvalue`值。

(4) 峰度（kurtosis）描述的是概率分布曲线的陡峭程度。我们来做一个峰度检验。该检验与偏度检验类似，当然这里是针对峰度。

```python
stats.kurtosistest(generated)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212230552246.png" alt="image-20250212230552246" style="zoom:50%;" />

(5) 正态性检验（normality test）可以检查数据集服从正态分布的程度。我们来做一个正态性检验。该检验同样有两个返回值，其中第二个返回值为p-value。

```python
stats.normaltest(generated)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212230647435.png" alt="image-20250212230647435" style="zoom:50%;" />

(6) 使用SciPy我们可以很方便地得到数据所在的区段中某一百分比处的数值：

```python
stats.scoreatpercentile(generated, 95)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212230735236.png" alt="image-20250212230735236" style="zoom:50%;" />

(7) 将前一步反过来，我们也可以从数值1出发找到对应的百分比：

```python
stats.percentileofscore(generated, 1)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212230820613.png" alt="image-20250212230820613" style="zoom:50%;" />

(8) 使用Matplotlib绘制生成数据的分布直方图。有关Matplotlib的详细介绍可以在前一章中 找到。

```python
import matplotlib.pyplot as plt
plt.hist(generated)
plt.show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212230926924.png" alt="image-20250212230926924" style="zoom:50%;" />

刚才做了些什么 : 我们按正态分布生成了一个随机数据集，并使用`scipy.stats`模块分析了该数据集。

完整代码：

```python
from scipy import stats
import matplotlib.pyplot as plt

generated = stats.norm.rvs(size = 900)
print(stats.norm.fit(generated))
print(stats.skewtest(generated))
print(stats.kurtosistest(generated))
print(stats.normaltest(generated))
print(stats.scoreatpercentile(generated, 95))
print(stats.percentileofscore(generated, 1))
plt.hist(generated)
plt.show()
```



## 勇敢出发：改进数据生成

从本节中的直方图来看，数据生成仍有改进的空间。尝试使用NumPy或调节`scipy.stats.norm.rvs`函数的参数。

直方图看起来像一个 **标准正态分布**（均值=0，标准差=1），但可以尝试一些改进：

1. **增加样本数量** 以使分布更加平滑。
2. **使用 `numpy.random.normal` 生成数据**，并尝试调整均值 (`loc`) 和标准差 (`scale`)。
3. **调整 `scipy.stats.norm.rvs` 的 `loc` 和 `scale` 参数**，生成不同的正态分布。

### **优化方案 1：增加样本数量**

当前 `size=900`，可以增加到 **5000 或更多** 以获得更平滑的分布：

```python
generated = stats.norm.rvs(size=5000)  # 增加样本数量
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212231440453.png" alt="image-20250212231440453" style="zoom:50%;" />

### **优化方案 2：使用 NumPy 生成数据**

使用 `numpy.random.normal()` 代替 `stats.norm.rvs()`，并调整 **均值 (`loc`) 和 标准差 (`scale`)**：

```python
generated = np.random.normal(loc=0, scale=1, size=5000)  # 调整均值和标准差
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212231542313.png" alt="image-20250212231542313" style="zoom:50%;" />

### **优化方案 3：调整 `scipy.stats.norm.rvs` 的参数**

你可以尝试 **不同的 `loc` 和 `scale` 值** 来调整分布的形状：

```python
generated = stats.norm.rvs(loc=0.5, scale=1.2, size=5000)  # 右偏移，增加方差
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212231618321.png" alt="image-20250212231618321" style="zoom:50%;" />

参考完整代码：

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 指定黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 "-" 显示为方块的问题
# 生成更平滑的正态分布数据
generated = np.random.normal(loc=0, scale=1, size=5000)

# 统计分析
print("拟合参数 (均值, 标准差):", stats.norm.fit(generated))
print("偏度检验:", stats.skewtest(generated))
print("峰度检验:", stats.kurtosistest(generated))
print("正态性检验:", stats.normaltest(generated))
print("95% 分位数:", stats.scoreatpercentile(generated, 95))
print("得分1的百分位:", stats.percentileofscore(generated, 1))

# 绘制更平滑的直方图
plt.hist(generated, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')
plt.title("改进后的正态分布直方图")
plt.show()

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250212232052630.png" alt="image-20250212232052630" style="zoom:50%;" />

# 10.5 样本比对和 SciKits

我们经常会遇到两组数据样本，它们可能来自不同的实验，但互相有一些关联。统计检验可以进行样本比对。

`scipy.stats`模块中已经实现了部分统计检验。

另一种笔者喜欢的统计检验是`scikits.statsmodels.stattools`中的`Jarque-Bera`正态性检验。

SciKits是Python的小型实验工具包，它并不是SciPy的一部分。

此外还有pandas（Python Data Analysis Library），它是scikits.statsmodels的分支。

你可以访问https://scikits.appspot.com/scikits查阅SciKits的模块索引。

你可以使用setuptools安装statsmodels，命令如下：

```python
easy_install statsmodels 
```

# 10.6 动手实践：比较股票对数收益率

我们将使用Matplotlib下载一年以来的两只股票的数据。如同前面的章节中所述，我们可以从雅虎财经频道获取股价数据。我们将比较DIA和SPY收盘价的对数收益率。我们还将在两只股票对数收益率的差值上应用`Jarque-Bera`正态性检验。请完成如下步骤。

(1) 编写一个函数`get_close`，用于返回指定股票的收盘价数据。

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf

def get_close(symbol):
    # 获取今天的日期
    today = date.today()

    # 获取去年的日期作为开始日期
    start = (today.year - 1, today.month, today.day)

    # 使用 yfinance 获取历史股市数据
    data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
    # display(data)
    # 提取收盘价和成交量并确保它们是一维数组
    close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
    return close
```

例如：`AAPL`代表苹果的股票：

```python
get_close('AAPL')
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213141637665.png" alt="image-20250213141637665" style="zoom:50%;" />

经过测试，书上的DIA和SPY都是可以获取收盘价的。目前这两只股票是有效的。

(2) 计算DIA和SPY的对数收益率。先对收盘价取自然对数，然后计算连续值之间的差值，即得到对数收益率。

```python
dia = np.diff(np.log(get_close('DIA')))
spy = np.diff(np.log(get_close('SPY')))
```

(3) 均值检验可以检查两组不同的样本是否有相同的均值。返回值有两个，其中第二个为p-value，取值范围为0~1。

```python
stats.ttest_ind(dia, spy)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213142320110.png" style="zoom:50%;" />

因此有77.7%的概率两组样本对数收益率的均值相同。

## 补充：`scipy.stats.ttest_ind`

`scipy.stats.ttest_ind` 是 SciPy 库中的一个函数，用于进行 **独立样本 t 检验**（Independent T-test），通常用于比较两个独立样本的均值是否存在显著差异。

```python
scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
```

参数：

`a` : array_like

- 第一个样本的数据（1D 或 2D 数组）。它可以是一个包含多个观测值的一维数组，也可以是多个组的二维数组。

`b` : array_like

- 第二个样本的数据，与 `a` 参数相同，表示第二组样本。

`axis` : int, optional, default 0

- 指定沿哪个轴进行操作。如果输入是二维数据，`axis=0` 表示在每列进行比较；`axis=1` 表示在每行进行比较。

`equal_var` : bool, optional, default True

- 是否假设两个样本具有相同的方差。默认为 True，表示进行 **标准的 t 检验**。如果设为 False，则进行 **Welch’s t-test**，该检验不假定两个样本具有相同的方差。

`nan_policy` : {'propagate', 'raise', 'omit'}, optional, default 'propagate'

- 处理 NaN 值的策略：
    - `'propagate'`: 如果输入包含 NaN，返回 NaN。
    - `'raise'`: 如果输入包含 NaN，抛出 `ValueError` 异常。
    - `'omit'`: 忽略 NaN 值，计算时不包括包含 NaN 的值。

返回值：该函数返回两个值：

- `statistic`: t 值，表示检验统计量。
- `pvalue`: p 值，用于判断均值差异是否显著。如果 p 值小于显著性水平（例如 0.05），则可以拒绝原假设，认为两个样本的均值存在显著差异。

---

(4) Kolmogorov-Smirnov检验可以判断两组样本同分布的可能性。

```python
print("判断两组样本同分布", stats.ks_2samp(spy, dia))
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213142533040.png" alt="image-20250213142533040" style="zoom:50%;" />

同样，该函数有两个返回值，其中第二个为p-value。

----

## 补充：`stats.ks_2samp`

`scipy.stats.ks_2samp` 是 SciPy 库中的一个函数，用于进行 **Kolmogorov-Smirnov 双样本检验**（KS 2-sample test）。该检验用于比较两个独立样本是否来自相同的分布。它基于两个样本的经验分布函数（ECDF，Empirical Cumulative Distribution Function）之间的最大差异。

```python
scipy.stats.ks_2samp(x1, x2, alternative='two-sided', method='auto')
```

参数：

**x1** : array_like

- 第一个样本的数据。可以是 1D 数组或类似的序列类型。

**x2** : array_like

- 第二个样本的数据。类型同 `x1`。

**alternative** : {'two-sided', 'less', 'greater'}, optional, default 'two-sided'

- 检验的方向：
    - `'two-sided'`: 双尾检验（默认），检验两个样本是否来自不同的分布。
    - `'less'`: 单尾检验，检验 `x1` 的分布是否“较小”。
    - `'greater'`: 单尾检验，检验 `x1` 的分布是否“较大”。

**method** : {'auto', 'exact', 'asymp'}, optional, default 'auto'

- 用于计算检验统计量的方法：
    - `'auto'`: 根据样本大小自动选择方法。
    - `'exact'`: 使用精确的方法进行计算。
    - `'asymp'`: 使用渐近方法（对于大样本有效）。

返回值：

**statistic** : float

- Kolmogorov-Smirnov 统计量，表示两个样本的经验分布函数之间的最大差异。

**pvalue** : float

- p 值，表示在原假设下，得到当前或更极端统计量的概率。较小的 p 值表示拒绝原假设，即认为两个样本的分布有显著差异。

----

(5) 在两只股票对数收益率的差值上应用Jarque-Bera正态性检验。

```python
print("正态性检验：", stats.jarque_bera(spy - dia)[1])
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213143532102.png" alt="image-20250213143532102" style="zoom:50%;" />

**非常小的 p 值**（比如上面的 `1.09e-17`）表明，**拒绝原假设的证据非常强**。也就是说，样本数据不符合正态分布的可能性极大。可能样本数据的偏度（skewness）和峰度（kurtosis）与正态分布存在较大差异。**Jarque-Bera 检验** 结合了样本的 **偏度** 和 **峰度**，如果这两个值偏离正态分布的期望（0 的偏度，3 的峰度），检验统计量会较大，从而导致 p 值非常小。

## 补充：`scikits.stats.jarque_bera`

安装`scikits`:

```python
import sys
!{sys.executable} -m pip install scikits.stats
```

安装会报错：

```python
Defaulting to user installation because normal site-packages is not writeable
ERROR: Could not find a version that satisfies the requirement scikits.stats (from versions: none)
ERROR: No matching distribution found for scikits.stats
```

由于 `scikits.stats` 库不再被维护和更新，因此无法通过 `pip install scikits.stats` 安装它。

**`scikits.stats`** 库的 **`jarque_bera`** 检验已经被移除或者不再更新，且没有提供新的版本用于安装。这是一个不再维护的第三方库，不能通过 pip 安装。

解决：**使用 SciPy 的 `jarque_bera` 检验：** 当前 **SciPy** 已经包含了 **Jarque-Bera 检验**，你可以直接使用 `scipy.stats.jarque_bera` 函数来进行正态性检验。这样你就不需要依赖 `scikits.stats` 库了。

---

`scipy.stats.jarque_bera` 是 SciPy 库中的一个函数，用于进行 **Jarque-Bera 正态性检验**，它是检验样本数据是否符合正态分布的一种方法。该检验基于样本的偏度（skewness）和峰度（kurtosis），并计算出一个统计量和对应的 p 值。

函数签名：

```python
scipy.stats.jarque_bera(x)
```

参数：**x** : array_like

- 要检验的样本数据，通常是一个一维数组或者列表。

返回值：

**statistic** : float

- Jarque-Bera 检验的统计量，表示数据的偏度和峰度偏差。

**pvalue** : float

- p 值，表示在原假设下，得到当前或更极端的统计量的概率。较小的 p 值（如小于 0.05）表明数据可能不符合正态分布。

---

(6) 使用Matplotlib绘制对数收益率以及其差值的直方图。

```python
plt.hist(spy, histtype="step", lw=1, label="SPY")  
plt.hist(dia, histtype="step", lw=2, label="DIA")  
plt.hist(spy - dia, histtype="step", lw=3,label="Delta")  
plt.legend()  
plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213143925099.png" alt="image-20250213143925099" style="zoom:50%;" />

刚才做了些什么 ：我们比较了DIA和SPY样本数据的对数收益率，还对它们的差值应用了Jarque-Bera正态性检验。

完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf
from scipy import stats

def get_close(symbol):
    # 获取今天的日期
    today = date.today()

    # 获取去年的日期作为开始日期
    start = (today.year - 1, today.month, today.day)

    # 使用 yfinance 获取历史股市数据
    data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
    # display(data)
    # 提取收盘价和成交量并确保它们是一维数组
    close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
    return close

dia = np.diff(np.log(get_close('DIA')))
spy = np.diff(np.log(get_close('SPY')))
print("t检验：",stats.ttest_ind(dia, spy))
print("判断两组样本同分布", stats.ks_2samp(spy, dia))
print("正态性检验：", stats.jarque_bera(spy - dia)[1])

# 绘图
plt.hist(spy, histtype="step", lw=1, label="SPY")  
plt.hist(dia, histtype="step", lw=2, label="DIA")  
plt.hist(spy - dia, histtype="step", lw=3,label="Delta")  
plt.legend()  
plt.show() 
```

---

# 10.7 信号处理

`scipy.signal`模块中包含滤波函数和B样条插值（B-spline interpolation）函数。

> 样条插值使用称为样条的多项式进行插值。插值过程将分段多项式连接起来拟合数据。B样条是样条的一种类型。

## 补充：样条插值

样条插值：一种以 可变样条 来作出一条经过一系列点的光滑曲线的数学方法。插值样条是由一些多项式组成的，每一个多项式都是由相邻的两个数据点决定的，这样，任意的两个相邻的多项式以及它们的导数在连接点处都是连续的。

简单理解，就是每两个点之间确定一个函数，这个函数就是一个样条，函数不同，样条就不同。所以定义中说 可变样条，然后把所有样条分段结合成一个函数，就是最终的插值函数。

### 线性样条

两点确定一条直线，我们可以在每两点间画一条直线，就可以把所有点连起来。显然曲线不够光滑，究其原因是因为连接点处导数不相同。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213145910580.png" alt="image-20250213145910580" style="zoom:50%;" />

### 二次样条

直线不行，用曲线代替，二次函数是最简单的曲线。

假设4个点，$$x_0，x_1，x_2，x_3$$，有3个区间，需要3个二次样条，每个二次样条为 $$ax^2+bx+c$$，故总计9个未知数。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213150009697.png" alt="image-20250213150009697" style="zoom:50%;" />

1. x0，x3两个端点都有一个二次函数经过，可确定2个方程

2. x1，x2两个中间点都有两个二次函数经过，可确定4个方程

3. 中间点处必须连续，需要保证左右二次函数一阶导相等，即　$$2*a_1*x_1+b_1=2*a_2*x_1+b_2$$和$$2*a_2*x_2+b_2=2*a_3*x_2+b_3$$，可确定2个方程，此时共有了8个方程。

4. 这里假设第一方程的二阶导为0，即 $$a_1=0$$，又是一个方程，共计9个方程。　　　

   > 为什么？端点可以有多种不同的限制，常见有3种。
   >
   > - 自由边界 Natural
   >
   > 首尾两端没有受到任何使他们弯曲的力，二次样条就是 $$s’=0$$，三次样条就是 $$s''=0$$。这种边界条件简洁且自然，通常用于描述物理上没有约束的情况。就像一个柔软的线条，两端没有被固定或拉紧，所以两端的弯曲是自由的。
   >
   > - 固定边界 Clamped
   >
   > 首尾两端点的微分值被指定，这种边界条件比较强，适用于需要控制端点斜率的情况，比如力学问题中的物体位置和运动的约束。
   >
   > - 非节点边界 Not-A-Knot
   >
   > 把端点当做中间点处理，三次函数不做假设，即
   >
   > <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213151359842.png" alt="image-20250213151359842" style="zoom:50%;" />
   >
   > 即 **端点与相邻内点之间的连续性不做额外假设**。在三次样条中，非节点边界要求两个端点之间的 **第三阶导数** 也连续。
   >
   > 上面假设$$a_1 = 0$$，表示该区间的样条是线性的（即没有二次项），这可以与 **自然边界条件** 中 **二阶导数为零** 的情形相关。这相当于对样条函数的平滑性进行约束，并且它为你的方程组增加了 **1 个额外的约束**，使得你有 **9 个方程**，正好可以求解 **9 个未知数**。简单理解，就是8个方程解9个未知数解不了，加个方程就可以了，此时最简单的假设就是$$a_1 = 0$$。

   上面9个方程联立即可求解9个未知数。

   <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213152700596.png" style="zoom:50%;" />

   二次样条插值连续光滑，只是前两个点之间是条直线，因为假设$$a_1 = 0$$，二次函数变成$$b_1x+c_1$$，显然是直线；

   而且最后两个点之间过于陡峭 。

   #### 代码

   ```python
   # encoding:utf-8
   import numpy as np
   import matplotlib.pyplot as plt
   from pylab import mpl
   """
   二次样条实现
   """
   x = [3, 4.5, 7, 9]
   y = [2.5, 1, 2.5, 0.5]
   
   # 解决中文显示问题
   plt.rcParams['font.sans-serif'] = ['STHeiti']  # 指定华文黑体
   plt.rcParams['axes.unicode_minus'] = False   # 解决负号 "-" 显示为方块的问题
   
   def calculateEquationParameters(x):
       #parameter为二维数组，用来存放参数，sizeOfInterval是用来存放区间的个数
       parameter = []
       sizeOfInterval=len(x)-1
       i = 1
       #首先输入方程两边相邻节点处函数值相等的方程为2n-2个方程
       while i < len(x)-1:
           data = init(sizeOfInterval*3)
           data[(i-1)*3]=x[i]*x[i]
           data[(i-1)*3+1]=x[i]
           data[(i-1)*3+2]=1
           data1 =init(sizeOfInterval*3)
           data1[i * 3] = x[i] * x[i]
           data1[i * 3 + 1] = x[i]
           data1[i * 3 + 2] = 1
           temp=data[1:]
           parameter.append(temp)
           temp=data1[1:]
           parameter.append(temp)
           i += 1
       #输入端点处的函数值。为两个方程,加上前面的2n-2个方程，一共2n个方程
       data = init(sizeOfInterval*3-1)
       data[0] = x[0]
       data[1] = 1
       parameter.append(data)
       data = init(sizeOfInterval *3)
       data[(sizeOfInterval-1)*3+0] = x[-1] * x[-1]
       data[(sizeOfInterval-1)*3+1] = x[-1]
       data[(sizeOfInterval-1)*3+2] = 1
       temp=data[1:]
       parameter.append(temp)
       #端点函数值相等为n-1个方程。加上前面的方程为3n-1个方程,最后一个方程为a1=0总共为3n个方程
       i=1
       while i < len(x) - 1:
           data = init(sizeOfInterval * 3)
           data[(i - 1) * 3] =2*x[i]
           data[(i - 1) * 3 + 1] =1
           data[i*3]=-2*x[i]
           data[i*3+1]=-1
           temp=data[1:]
           parameter.append(temp)
           i += 1
       return parameter
   
   """
   对一个size大小的元组初始化为0
   """
   def init(size):
       j = 0
       data = []
       while j < size:
           data.append(0)
           j += 1
       return data
   
   
   """
   功能：计算样条函数的系数。
   参数：parametes为方程的系数，y为要插值函数的因变量。
   返回值：二次插值函数的系数。
   """
   
   def solutionOfEquation(parametes,y):
       sizeOfInterval = len(x) - 1
       result = init(sizeOfInterval*3-1)
       i=1
       while i<sizeOfInterval:
           result[(i-1)*2]=y[i]
           result[(i-1)*2+1]=y[i]
           i+=1
       result[(sizeOfInterval-1)*2]=y[0]
       result[(sizeOfInterval-1)*2+1]=y[-1]
       a = np.array(calculateEquationParameters(x))
       b = np.array(result)
       return np.linalg.solve(a,b)
   
   """
   功能：根据所给参数，计算二次函数的函数值：
   参数:parameters为二次函数的系数，x为自变量
   返回值：为函数的因变量
   """
   def calculate(paremeters,x):
       result=[]
       for data_x in x:
           result.append(paremeters[0]*data_x*data_x+paremeters[1]*data_x+paremeters[2])
       return  result
   
   
   """
   功能：将函数绘制成图像
   参数：data_x,data_y为离散的点.new_data_x,new_data_y为由拉格朗日插值函数计算的值。x为函数的预测值。
   返回值：空
   """
   def  Draw(data_x,data_y,new_data_x,new_data_y):
           plt.plot(new_data_x, new_data_y, label=u"拟合曲线", color="black")
           plt.scatter(data_x,data_y, label=u"离散数据",color="red")
           mpl.rcParams['font.sans-serif'] = ['STHeiti']
           mpl.rcParams['axes.unicode_minus'] = False
           plt.title(u"二次样条函数")
           plt.legend(loc="upper left")
           plt.show()
   
   result=solutionOfEquation(calculateEquationParameters(x),y)
   new_data_x1=np.arange(3, 4.5, 0.1)
   new_data_y1=calculate([0,result[0],result[1]],new_data_x1)
   new_data_x2=np.arange(4.5, 7, 0.1)
   new_data_y2=calculate([result[2],result[3],result[4]],new_data_x2)
   new_data_x3=np.arange(7, 9.5, 0.1)
   new_data_y3=calculate([result[5],result[6],result[7]],new_data_x3)
   new_data_x=[]
   new_data_y=[]
   new_data_x.extend(new_data_x1)
   new_data_x.extend(new_data_x2)
   new_data_x.extend(new_data_x3)
   new_data_y.extend(new_data_y1)
   new_data_y.extend(new_data_y2)
   new_data_y.extend(new_data_y3)
   Draw(x,y,new_data_x,new_data_y)
   ```

   <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213152700596.png" alt="image-20250213152700596" style="zoom:50%;" />

### 三次样条

二次函数最高项系数为0，导致变成直线，那三次函数最高项系数为0，还是曲线，插值效果应该更好。

三次样条思路与二次样条基本相同，

同样假设4个点，$$x_0，x_1，x_2，x_3$$，有3个区间，需要3个三次样条，每个三次样条为 $$ax^3+bx^2+cx+d$$，故总计12个未知数。

1. 内部节点处的函数值应该相等，这里一共是4个方程。

2. 函数的第一个端点和最后一个端点，应该分别在第一个方程和最后一个方程中。这里是2个方程。

3. 两个函数(即中间两个点)在节点处的一阶导数应该相等。这里是两个方程。

4. 两个函数(即中间两个点)在节点处的二阶导数应该相等，这里是两个方程。　　　　 

5. 假设端点处的二阶导数（二阶导是$$3ax^2+2bx+c$$）为零，这里是两个方程。　　　$$a_1=0$$ ，$$b_1=0$$

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213153333445.png" alt="image-20250213153333445" style="zoom:50%;" />

#### 代码

```python
# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
"""
三次样条实现
"""
x = [3, 4.5, 7, 9]
y = [2.5, 1, 2.5, 0.5]

def calculateEquationParameters(x):
    #parameter为二维数组，用来存放参数，sizeOfInterval是用来存放区间的个数
    parameter = []
    sizeOfInterval=len(x)-1;
    i = 1
    #首先输入方程两边相邻节点处函数值相等的方程为2n-2个方程
    while i < len(x)-1:
        data = init(sizeOfInterval*4)
        data[(i-1)*4] = x[i]*x[i]*x[i]
        data[(i-1)*4+1] = x[i]*x[i]
        data[(i-1)*4+2] = x[i]
        data[(i-1)*4+3] = 1
        data1 =init(sizeOfInterval*4)
        data1[i*4] =x[i]*x[i]*x[i]
        data1[i*4+1] =x[i]*x[i]
        data1[i*4+2] =x[i]
        data1[i*4+3] = 1
        temp = data[2:]
        parameter.append(temp)
        temp = data1[2:]
        parameter.append(temp)
        i += 1
    # 输入端点处的函数值。为两个方程, 加上前面的2n - 2个方程，一共2n个方程
    data = init(sizeOfInterval * 4 - 2)
    data[0] = x[0]
    data[1] = 1
    parameter.append(data)
    data = init(sizeOfInterval * 4)
    data[(sizeOfInterval - 1) * 4 ] = x[-1] * x[-1] * x[-1]
    data[(sizeOfInterval - 1) * 4 + 1] = x[-1] * x[-1]
    data[(sizeOfInterval - 1) * 4 + 2] = x[-1]
    data[(sizeOfInterval - 1) * 4 + 3] = 1
    temp = data[2:]
    parameter.append(temp)
    # 端点函数一阶导数值相等为n-1个方程。加上前面的方程为3n-1个方程。
    i=1
    while i < sizeOfInterval:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = 3 * x[i] * x[i]
        data[(i - 1) * 4 + 1] = 2 * x[i]
        data[(i - 1) * 4 + 2] = 1
        data[i * 4] = -3 * x[i] * x[i]
        data[i * 4 + 1] = -2 * x[i]
        data[i * 4 + 2] = -1
        temp = data[2:]
        parameter.append(temp)
        i += 1
    # 端点函数二阶导数值相等为n-1个方程。加上前面的方程为4n-2个方程。且端点处的函数值的二阶导数为零，为两个方程。总共为4n个方程。
    i = 1
    while i < len(x) - 1:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = 6 * x[i]
        data[(i - 1) * 4 + 1] = 2
        data[i * 4] = -6 * x[i]
        data[i * 4 + 1] = -2
        temp = data[2:]
        parameter.append(temp)
        i += 1
    return parameter



"""
对一个size大小的元组初始化为0
"""
def init(size):
    j = 0
    data = []
    while j < size:
        data.append(0)
        j += 1
    return data

"""
功能：计算样条函数的系数。
参数：parametes为方程的系数，y为要插值函数的因变量。
返回值：三次插值函数的系数。
"""
def solutionOfEquation(parametes,y):
    sizeOfInterval = len(x) - 1
    result = init(sizeOfInterval*4-2)
    i=1
    while i<sizeOfInterval:
        result[(i-1)*2]=y[i]
        result[(i-1)*2+1]=y[i]
        i+=1
    result[(sizeOfInterval-1)*2]=y[0]
    result[(sizeOfInterval-1)*2+1]=y[-1]
    a = np.array(calculateEquationParameters(x))
    b = np.array(result)
    for data_x in b:
        print(data_x)
    return np.linalg.solve(a,b)

"""
功能：根据所给参数，计算三次函数的函数值：
参数:parameters为二次函数的系数，x为自变量
返回值：为函数的因变量
"""
def calculate(paremeters,x):
    result=[]
    for data_x in x:
        result.append(paremeters[0]*data_x*data_x*data_x+paremeters[1]*data_x*data_x+paremeters[2]*data_x+paremeters[3])
    return  result


"""
功能：将函数绘制成图像
参数：data_x,data_y为离散的点.new_data_x,new_data_y为由拉格朗日插值函数计算的值。x为函数的预测值。
返回值：空
"""
def  Draw(data_x,data_y,new_data_x,new_data_y):
        plt.plot(new_data_x, new_data_y, label=u"拟合曲线", color="black")
        plt.scatter(data_x,data_y, label=u"离散数据",color="red")
        mpl.rcParams['font.sans-serif'] = ['STHeiti']
        mpl.rcParams['axes.unicode_minus'] = False
        plt.title(u"三次样条函数")
        plt.legend(loc="upper left")
        plt.show()


result=solutionOfEquation(calculateEquationParameters(x),y)
new_data_x1=np.arange(3, 4.5, 0.1)
new_data_y1=calculate([0,0,result[0],result[1]],new_data_x1)
new_data_x2=np.arange(4.5, 7, 0.1)
new_data_y2=calculate([result[2],result[3],result[4],result[5]],new_data_x2)
new_data_x3=np.arange(7, 9.5, 0.1)
new_data_y3=calculate([result[6],result[7],result[8],result[9]],new_data_x3)
new_data_x=[]
new_data_y=[]
new_data_x.extend(new_data_x1)
new_data_x.extend(new_data_x2)
new_data_x.extend(new_data_x3)
new_data_y.extend(new_data_y1)
new_data_y.extend(new_data_y2)
new_data_y.extend(new_data_y3)
Draw(x,y,new_data_x,new_data_y)
```

-----

SciPy中以一组数值来定义信号。我们以detrend函数作为滤波器的一个例子。该函数可以对信号进行线性拟合，然后从原始输入数据中去除这个线性趋势。

# 10.8 动手实践：检测 QQQ 股价的线性趋势

相比于去除数据样本的趋势，我们通常更关心的是趋势本身。在去除趋势的操作之后，我们仍然很容易获取该趋势。我们将对QQQ一年以来的股价数据进行这些处理分析。

(1) 编写代码获取QQQ的收盘价和对应的日期数据。

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
symbol = 'QQQ'

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价并确保它们是一维数组
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
display(dates)
display(close)
```

(2) 去除信号中的线性趋势。

```python
from scipy import signal

y = signal.detrend(close)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213154348790.png" alt="image-20250213154348790" style="zoom:50%;" />

(3) 创建月定位器和日定位器。

```python
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator

alldays = DayLocator()  
months = MonthLocator () 
```

(4) 创建一个日期格式化器以格式化x轴上的日期。该格式化器将创建一个字符串，包含简写的月份和年份。

```python
month_formatter = DateFormatter("%b %Y") 
```

(5) 创建图像和子图。

```python
fig = plt.figure()
ax = fig.add_subplot(111)
```

(6) 绘制股价数据以及将去除趋势后的信号从原始数据中减去所得到的潜在趋势。

```python
plt.plot(dates, close, 'o', dates, close - y, '-')
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213154806807.png" alt="image-20250213154806807" style="zoom:50%;" />

(7) 设置定位器和格式化器。

```python
ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_major_formatter(month_formatter) 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213154839926.png" alt="image-20250213154839926" style="zoom:50%;" />

(8) 将x轴上的标签格式化为日期。

```python
# 自动旋转日期标签
fig.autofmt_xdate()

# 显示图形
plt.show()
```

最终效果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213154921159.png" alt="image-20250213154921159" style="zoom:50%;" />

刚才做了些什么 :  我们绘制了QQQ的收盘价数据以及对应的趋势线。

完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf
from scipy import signal
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

# 使用有效的股票符号（例如苹果公司AAPL）
symbol = 'QQQ'

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价并确保它们是一维数组
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
# display(dates)
# display(close)
 
# 去除信号中的线性趋势
y = signal.detrend(close)

# 定位器
alldays = DayLocator()  
months = MonthLocator () 
# 格式化器
month_formatter = DateFormatter("%b %Y") 

# 创建图像和子图
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(dates, close, 'o', dates, close - y, '-')

ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_major_formatter(month_formatter) 

# 自动旋转日期标签
fig.autofmt_xdate()

# 显示图形
plt.show()
```

# 10.9 傅里叶分析

现实世界中的信号往往具有周期性。傅里叶变换（Fourier transform）是处理这些信号的常用工具。傅里叶变换是一种从时域到频域的变换，也就是将周期信号线性分解为不同频率的正弦和余弦函数。

傅里叶变换的函数可以在`scipy.fftpack`模块中找到（NumPy也有自己的傅里叶工具包，即`numpy.fft`）。这个模块包含快速傅里叶变换、微分算子和拟微分算子以及一些辅助函数。MATLAB用户会很高兴，因为`scipy.fftpack`模块中的很多函数与MATLAB对应的函数同名，且功能也很相近。

> 注：在 `scipy` 中，傅立叶变换相关的模块是 `scipy.fft`，而不是 `scipy.fftpack`。
>
> 具体来说，`scipy.fftpack` 是旧版的傅立叶变换模块，已经被 `scipy.fft` 取代。`scipy.fft` 提供了更现代和高效的接口，并且是推荐使用的模块。
>
> 在代码中使用的是 `scipy.fftpack`，建议迁移到 `scipy.fft`，因为后者在新版本的 `scipy` 中得到了更好的支持和优化。

# 10.10 动手实践：对去除趋势后的信号进行滤波处理

在10.8节我们学习了如何去除信号中的趋势。

去除趋势后的信号可能有周期性的分量，我们将其显现出来。一些步骤已在前面的“动手实践”教程中出现过，如下载数据和设置Matplotlib 对象。这些步骤将被略去。

(1) 应用傅里叶变换，得到信号的频谱。

```python
amps = np.abs(fft.fftshift(fft.rfft(y))) 
```

(2) 滤除噪声。如果某一频率分量的大小低于最强分量的10%，则将其滤除。

```python
amps[amps < 0.1 * amps.max()] = 0 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213160814748.png" alt="image-20250213160814748" style="zoom:50%;" />

(3) 将滤波后的信号变换回时域，并和去除趋势后的信号一起绘制出来。

```python
plt.plot(dates, y, 'o', label="detrended") 
plt.plot(dates, -fft.irfft(fft.ifftshift(amps)), label="filtered") 
```

那么，效果就是：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213163639544.png" alt="image-20250213163639544" style="zoom:50%;" />

此时，还没有自动旋转日期标签，所以日期显示是这样。

(4) 将x轴上的标签格式化为日期，并添加一个特大号的图例。

```python
# 自动旋转日期标签
fig.autofmt_xdate()
# 添加一个特大的图例
plt.legend(prop={'size':'x-large'})
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213163746326.png" alt="image-20250213163746326" style="zoom:50%;" />

(5) 添加第二个子图，绘制滤波后的频谱。

```python
ax2 = fig.add_subplot(212) 
ax2.tick_params(axis='both', which='major', labelsize='x-large')  
N = len(close) 
plt.plot(np.linspace(-N/2, N/2, N), amps, label="transformed") 
```

此时报错了：

```python
ValueError: x and y must have same first dimension, but have shapes (251,) and (126,)
```

这个错误是由于 `plt.plot(np.linspace(-N/2, N/2, N), amps, label="transformed")` 中，`amps` 和 `np.linspace(-N/2, N/2, N)` 的长度不一致。具体来说，`amps` 的长度是通过 `rfft` 获得的，而 `rfft` 的输出频谱是比输入信号的长度少一半的（加一）。

解决：

1. **`amps` 和频率轴的长度不匹配**：由于使用的是 `rfft`，其结果长度为 `len(y) // 2 + 1`。需要调整频率轴，使它与 `amps` 的长度一致。
2. **频率轴的长度**：在绘制频谱图时，`amps` 和频率轴（`np.linspace(-N/2, N/2, N)`）的长度应当一致。

修改后代码：

```python
# N = len(close) 
# plt.plot(np.linspace(-N/2, N/2, N), amps, label="transformed") 
# 计算频率轴，并确保它和amps的长度一致
N = len(y)  # 使用y的长度来生成频率轴
frequencies = np.linspace(0, 1, len(amps)) * (N // 2)  # 从0到Nyquist频率
```

然后绘制第二幅图，即频谱图：

```python
# 绘制频谱图
plt.plot(frequencies, amps, label="transformed")

plt.legend(prop={'size':'x-large'})  
plt.show() 
```

效果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213165144177.png" alt="image-20250213165144177" style="zoom:50%;" />

> 不建议第一幅图`labelsize`太大，否则横轴的日期信息会被遮挡。

最终完整代码：

```python
from scipy import fft,signal
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import yfinance as yf
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y") 

# 使用有效的股票符号
symbol = 'QQQ'

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价并确保它们是一维数组
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
# print(len(dates))
# print(len(close))
 
# 去除信号中的线性趋势
y = signal.detrend(close)


fig = plt.figure()
fig.subplots_adjust(hspace=.3)  
ax = fig.add_subplot(211) 


ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_major_formatter(month_formatter) 

# 调大字号 
ax.tick_params(axis='both', which='major', labelsize='x-small') 

# 傅里叶变换，得到信号的频谱
amps = np.abs(fft.fftshift(fft.rfft(y))) 
# 滤除噪声。如果某一频率分量的大小低于最强分量的10%，则将其滤除。 
amps[amps < 0.1 * amps.max()] = 0 

# 将滤波后的信号变换回时域，并和去除趋势后的信号一起绘制出来。  
plt.plot(dates, y, 'o', label="detrended") 
# plt.plot(dates, -fft.irfft(fft.ifftshift(amps)), label="filtered") 
plt.plot(dates_filtered, filtered_signal, label="filtered") 

# 自动旋转日期标签
fig.autofmt_xdate()
# 添加一个特大的图例
plt.legend(prop={'size':'x-large'})
# 添加第二个子图，绘制滤波后的频谱
ax2 = fig.add_subplot(212) 
ax2.tick_params(axis='both', which='major', labelsize='x-large')  
# N = len(close) 
# plt.plot(np.linspace(-N/2, N/2, N), amps, label="transformed") 
# 计算频率轴，并确保它和amps的长度一致
N = len(y)  # 使用y的长度来生成频率轴
frequencies = np.linspace(0, 1, len(amps)) * (N // 2)  # 从0到Nyquist频率

# 绘制频谱图
plt.plot(frequencies, amps, label="transformed")

plt.legend(prop={'size':'x-large'})  
plt.show() 
```

# 10.11 数学优化

优化算法（optimization algorithm）尝试寻求某一问题的最优解，例如找到函数的最大值或最小值，函数可以是线性或者非线性的。解可能有一些特定的约束，例如不允许有负数。在`scipy.optimize`模块中提供了一些优化算法，最小二乘法函数leastsq就是其中之一。当调用这个函数时，我们需要提供一个**残差（误差项）**函数。这样，leastsq将最小化残差的平方和。得到的解与我们使用的数学模型有关。我们还需要为算法提供一个起始点，这应该是一个最好的猜测——尽可能接近真实解。否则，程序执行800轮迭代后将停止。

# 10.12 动手实践：拟合正弦波

在10.10节中，我们为去除趋势后的数据创建了一个简单的滤波器。现在，我们使用一个限制性更强的滤波器，只保留主频率部分。我们将拟合一个正弦波并绘制结果。该模型有4个参数—— 振幅、频率、相位和垂直偏移。请完成如下步骤。

(1) 根据正弦波模型，定义residuals函数：

```python
import numpy as np

def residuals(p, y, x):  
    A, k, theta, b = p 
    err = y - A * np.sin(2 * np.pi * k * x + theta) + b  
    return err 
```

(2) 将滤波后的信号变换回时域：

```python
from scipy import fft

filtered = -fft.irfft(fft.ifftshift(amps)) 
```

(3) 猜测参数的值，尝试估计从时域到频域的变换函数：

```python
# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y") 

# 使用有效的股票符号
symbol = 'QQQ'

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价并确保它们是一维数组
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组

# 猜测参数的值，尝试估计从时域到频域的变换函数
N = len(close) 
f = np.linspace(-N/2, N/2, N) 
p0 = [filtered.max(), f[amps.argmax()]/(2*N), 0, 0]  
print("P0", p0 )
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250213165747993.png" alt="image-20250213165747993" style="zoom:50%;" />

(4) 调用leastsq函数：

```python
# 下面的步骤的意义：修改dates的类型，否则传入residuals函数，会无法和k相乘
# 将日期转换为 pandas 的 DatetimeIndex 类型
dates = pd.to_datetime(dates)
# 将日期转换为天数（相对于第一个日期）
date_numeric = (dates - dates[0]).days

# 确保 filtered 和 date_numeric 数组的长度一致,否则optimize.leastsq还是会报错
if len(filtered) > len(date_numeric):
    filtered = filtered[:len(date_numeric)]
elif len(date_numeric) > len(filtered):
    date_numeric = date_numeric[:len(filtered)]
print(filtered.shape,date_numeric.shape) # 输出：(250,) (250,)

# 调用leastsq函数
plsq = optimize.leastsq(residuals, p0, args=(filtered,date_numeric))  
p = plsq[0]  
print("P", p)
```

> 代码比书上的多，是因为现在依赖更新了，处理方式也会有区别。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214125015053.png" alt="image-20250214125015053" style="zoom:50%;" />

(5) 在第一个子图中绘制去除趋势后的数据、滤波后的数据及其拟合曲线。将x轴格式化为日期，并添加一个图例。

```python
alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y")

fig = plt.figure()  
fig.subplots_adjust(hspace=.3)  
ax = fig.add_subplot(211) 

ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.tick_params(axis='both', which='major', labelsize='x-large')
print(date_numeric)
plt.plot(dates_251, y, 'o', label="detrended")  
plt.plot(dates_, filtered, label="filtered") 
plt.plot(dates_, p[0] * np.sin(2 * np.pi * date_numeric * p[1] + p[2]) + p[3], '^', label="fit")  
fig.autofmt_xdate() 
plt.legend(prop={'size':'x-large'}) 
```

> 注，此处我又对数据（日期数据）进行了处理：
>
> ```python
> dates_251 = dates
> dates = dates[:len(filtered)]
> dates_ = dates  # 长度250
> # 下面的步骤的意义：修改dates的类型，否则传入residuals函数，会无法和k相乘
> # 将日期转换为 pandas 的 DatetimeIndex 类型
> dates = pd.to_datetime(dates)
> print(dates)
> # 将日期转换为天数（相对于第一个日期）
> date_numeric = (dates - dates[0]).days
> ```

(6) 添加第二个子图，绘制主频率部分的频谱图和图例。

```python
# 第2张子图
ax2 = fig.add_subplot(212) 
ax2.tick_params(axis='both', which='major', labelsize='x-large')  
# 这里调整 f 数组的长度与 amps 一致，否则画图会报错
f = np.fft.fftfreq(N, (dates[1] - dates[0]).days)[:len(amps)]
plt.plot(f, amps, label="transformed") 
plt.legend(prop={'size':'x-large'})  
plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214130735458.png" alt="image-20250214130735458" style="zoom:50%;" />

我建议把图的图例的`x-large`都去掉，否则第一幅子图的字都看不清，被遮挡了(或者改成`x-small`)，最终完整代码和效果如下：

```python
import numpy as np
from scipy import fft, optimize, signal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator
import yfinance as yf
from datetime import date


def residuals(p, y, x):  
    A, k, theta, b = p 
    err = y - A * np.sin(2 * np.pi * k * x + theta) + b  
    return err 

# 获取今天的日期
today = date.today()

# 获取去年的日期作为开始日期
start = (today.year - 1, today.month, today.day)

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y") 

# 使用有效的股票符号
symbol = 'QQQ'

# 使用 yfinance 获取历史股市数据
data = yf.download(symbol, start=f"{start[0]}-{start[1]}-{start[2]}", end=f"{today.year}-{today.month}-{today.day}")
# display(data)
# 提取日期和收盘价并确保它们是一维数组
dates = data.index.to_numpy().flatten()
close = data['Close'].to_numpy().flatten()  # 使用 flatten() 确保是一维数组
# print(len(dates))
# print(len(close))
 
# 去除信号中的线性趋势
y = signal.detrend(close)

# 傅里叶变换，得到信号的频谱
amps = np.abs(fft.fftshift(fft.rfft(y))) 
# 滤除噪声。如果某一频率分量的大小低于最强分量的10%，则将其滤除。 
amps[amps < 0.1 * amps.max()] = 0 

# 将滤波后的信号变换回时域
filtered = -fft.irfft(fft.ifftshift(amps)) 

# 猜测参数的值，尝试估计从时域到频域的变换函数
N = len(close) 
f = np.linspace(-N/2, N/2, N) 
p0 = [filtered.max(), f[amps.argmax()]/(2*N), 0, 0]  
print("P0", p0 )  # 输出：P0 [19.521577364265628, -0.11800000000000001, 0, 0]
dates_251 = dates
dates = dates[:len(filtered)]
dates_ = dates  # 长度250
# 下面的步骤的意义：修改dates的类型，否则传入residuals函数，会无法和k相乘
# 将日期转换为 pandas 的 DatetimeIndex 类型
dates = pd.to_datetime(dates)
# print(dates)
# 将日期转换为天数（相对于第一个日期）
date_numeric = (dates - dates[0]).days

# 调用leastsq函数
plsq = optimize.leastsq(residuals, p0, args=(filtered,date_numeric))  
p = plsq[0]  
print("P", p)

alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y")

fig = plt.figure()  
fig.subplots_adjust(hspace=.3)  
ax = fig.add_subplot(211) 

ax.xaxis.set_minor_locator(alldays) 
ax.xaxis.set_major_locator(months) 
ax.xaxis.set_major_formatter(month_formatter) 
ax.tick_params(axis='both', which='major', labelsize='x-small')
# print(date_numeric)
plt.plot(dates_251, y, 'o', label="detrended")  
plt.plot(dates_, filtered, label="filtered") 
plt.plot(dates_, p[0] * np.sin(2 * np.pi * date_numeric * p[1] + p[2]) + p[3], '^', label="fit")  
fig.autofmt_xdate() 
# plt.legend(prop={'size':'x-large'}) 

# 第2张子图
ax2 = fig.add_subplot(212) 
ax2.tick_params(axis='both', which='major', labelsize='x-small')  
# 这里调整 f 数组的长度与 amps 一致，否则画图会报错
f = np.fft.fftfreq(N, (dates[1] - dates[0]).days)[:len(amps)]
plt.plot(f, amps, label="transformed") 
# plt.legend(prop={'size':'x-large'})  
plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214131007498.png" alt="image-20250214131007498" style="zoom:50%;" />

刚才做了些什么 :  我们对一年以来的QQQ股价数据进行了去趋势处理。然后进行了滤波处理，仅保留了频谱上的主频率部分。我们使用scipy.optimize模块对滤波后的信号拟合了一个正弦波函数。

# 10.13 数值积分

SciPy中有数值积分的包`scipy.integrate`，在NumPy中没有相同功能的包。quad函数可以求单变量函数在两点之间的积分，这些点之间的距离可以是无穷小或无穷大。该函数使用最简单的数值积分方法即梯形法则（trapezoid rule）进行计算。

# 10.14 动手实践：计算高斯积分

高斯积分（Gaussian integral）出现在误差函数（数学中记为erf）的定义中，但高斯积分本身的积分区间是无穷的，它的值等于pi的平方根。我们将使用quad函数计算它。

例如，下面都是常见的高斯积分公式。（第一个相信我肯定见过,下面代码计算的应该也是第一个）

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214131243801.png" alt="image-20250214131243801" style="zoom:50%;" />

使用quad函数计算高斯积分。

```python
from scipy import integrate
import numpy as np

print("Gaussian integral", np.sqrt(np.pi))
print(integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf) )
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214131446907.png" alt="image-20250214131446907" style="zoom:50%;" />

刚才做了些什么 ：  我们使用quad函数计算了高斯积分。

# 10.15 插值

插值（interpolation）即在数据集已知数据点之间“填补空白”。`scipy.interpolate`函数可以根据实验数据进行插值。interp1d类可以创建线性插值（linear interpolation）或三次插值（cubic interpolation）的函数。默认将创建线性插值函数，三次插值函数可以通过设置kind参数来创建。interp2d类的工作方式相同，只不过用于二维插值。

# 10.16 动手实践：一维插值

我们将使用`sinc`函数创建数据点并添加一些随机噪音。随后，我们将进行线性插值和三次插值，并绘制结果。请完成如下步骤。

(1) 创建数据点并添加噪音：

```python
import numpy as np

x = np.linspace(-18, 18, 36)
noise = 0.1 * np.random.random(len(x)) 
signal = np.sinc(x) + noise 
```

(2) 创建一个线性插值函数，并应用于有5倍数据点个数的输入数组：

```python
from scipy import interpolate

interpreted = interpolate.interp1d(x, signal)
x2 = np.linspace(-18, 18, 180)  
y = interpreted(x2) 
```

(3) 执行与前一步相同的操作，不过这里使用三次插值。

```python
cubic = interpolate.interp1d(x, signal, kind="cubic")  
y2 = cubic(x2)
```

(4) 使用Matplotlitb绘制结果。

```python
import matplotlib.pyplot as plt

plt.plot(x, signal, 'o', label="data")  
plt.plot(x2, y, '-', label="linear")  
plt.plot(x2, y2, '-', lw=2, label="cubic") 
plt.legend()  
plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214131933449.png" alt="image-20250214131933449" style="zoom:50%;" />

刚才做了些什么 :  我们用sinc函数创建了一个数据集并加入了噪音，然后使用`scipy.interpolate`模块中的interp1d类进行了线性插值和三次插值。

完整代码如下：

```python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x = np.linspace(-18, 18, 36)
noise = 0.1 * np.random.random(len(x)) 
signal = np.sinc(x) + noise 
interpreted = interpolate.interp1d(x, signal)
x2 = np.linspace(-18, 18, 180)  
y = interpreted(x2) 
cubic = interpolate.interp1d(x, signal, kind="cubic")  
y2 = cubic(x2)
plt.plot(x, signal, 'o', label="data")  
plt.plot(x2, y, '-', label="linear")  
plt.plot(x2, y2, '-', lw=2, label="cubic") 
plt.legend()  
plt.show() 
```

# 10.17 图像处理

我们可以使用`scipy.ndimage`包进行图像处理。该模块包含各种图像滤波器和工具函数。

# 10.18 动手实践：处理 Lena 图像

在`scipy.misc`模块中，有一个函数可以载入Lena图像。这幅Lena Soderberg的图像是被用做图像处理的经典示例图像。我们将在该图像上应用一些滤波器，并进行旋转操作。请完成如下步骤。

> 注：`scipy.misc` 模块已经不再支持 `lena` 图像，它已经被弃用并从 SciPy 版本 1.3.0 开始移除。因此，`misc.lena()` 不再有效。
>
> 要解决这个问题，可以直接使用其他方式来加载 Lena 图像。可以使用 `matplotlib` 或 `PIL`（Python Imaging Library）来加载图像。
>
> 补充：[常见的图片库链接（往上随便找的）](https://www.eecs.qmul.ac.uk/~phao/IP/Images/)

(1) 载入Lena图像，并使用灰度颜色表将其在子图中显示出来。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# 从 URL 读取 Lena 图像
image_url = 'https://www.eecs.qmul.ac.uk/~phao/IP/Images/Lena.bmp'
image = io.imread(image_url)

# 将图像转换为 np.float32
image = image.astype(np.float32)

# 创建子图
plt.subplot(221)
plt.title("Original Image")

# 显示图像（灰度图）
img = plt.imshow(image, cmap=plt.cm.gray)

# 显示图像
plt.show()

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214133352031.png" alt="image-20250214133352031" style="zoom:50%;" />

注意，我们处理的是一个float32类型的数组。

(2) 中值滤波器扫描信号的每一个数据点，并替换为相邻数据点的中值。对图像应用中值滤波器并显示在第二个子图中。

```python
from scipy import ndimage

plt.subplot(222)  
plt.title("Median Filter") 
filtered = ndimage.median_filter(image, size=(42,42))  
plt.imshow(filtered, cmap=plt.cm.gray) 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214133513059.png" alt="image-20250214133513059" style="zoom:50%;" />

(3) 旋转图像并显示在第三个子图中。

```python
plt.subplot(223) 
plt.title("Rotated") 
rotated = ndimage.rotate(image, 90) 
plt.imshow(rotated, cmap=plt.cm.gray) 
```

(4) Prewitt滤波器是基于图像强度的梯度计算。对图像应用Prewitt滤波器并显示在第四个子图中。

```python
plt.subplot(224)  
plt.title("Prewitt Filter")  
filtered = ndimage.prewitt(image)  
plt.imshow(filtered, cmap=plt.cm.gray)  
plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214134141236.png" alt="image-20250214134141236" style="zoom:50%;" />

可以看到图片之间距离太小了，标注重叠了。为了解决图像标注重叠的问题，需要调整子图之间的间距。可以通过 `plt.subplots_adjust()` 来增加子图之间的空间。这个函数允许调整各个方向上的间距，比如左侧、右侧、上下间距等。

以下是修改后的完整代码和最终效果，增加了子图之间的间距：

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage

# 从 URL 读取 Lena 图像
image_url = 'https://www.eecs.qmul.ac.uk/~phao/IP/Images/Lena.bmp'
image = io.imread(image_url)

# 将图像转换为 np.float32
image = image.astype(np.float32)

# 创建子图
plt.subplot(221)
plt.title("Original Image")

# 显示图像（灰度图）
img = plt.imshow(image, cmap=plt.cm.gray)

# 显示图像
# plt.show()


plt.subplot(222)  
plt.title("Median Filter") 
filtered = ndimage.median_filter(image, size=(42,42))  
plt.imshow(filtered, cmap=plt.cm.gray) 

plt.subplot(223) 
plt.title("Rotated") 
rotated = ndimage.rotate(image, 90) 
plt.imshow(rotated, cmap=plt.cm.gray) 

plt.subplot(224)  
plt.title("Prewitt Filter")  
filtered = ndimage.prewitt(image)  
plt.imshow(filtered, cmap=plt.cm.gray)  

# 调整子图之间的间距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.5)

plt.show() 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214134253143.png" alt="image-20250214134253143" style="zoom:50%;" />

刚才做了些什么 : 我们使用`scipy.ndimage`模块对Lena图像进行了一些处理操作。

# 10.19 音频处理

既然我们已经完成了一些图像处理的操作，你可能不会惊讶我们也可以对WAV文件进行处理。我们将下载一个WAV文件并将其重复播放几次。下载音频的部分将被省略，只保留常规的Python代码。

# 10.20 动手实践：重复音频片段

我们将下载一个WAV文件，来自电影《王牌大贱谍》（Austin Powers）中的一声呼喊：“Smashing，baby!”使用`scipy.io.wavfile`模块中的read函数可以将该文件转换为一个NumPy 数组。在本节教程的最后，我们将使用同一模块中的write函数写入一个新的WAV文件。我们将使用tile函数来重复播放音频片段。请完成如下步骤。

> 注意原书的一些依赖已经失效了(包括以下函数的处理细节)。下面展示的是我2025-02-14测试可行的。

(1) 使用read函数读入文件：

```python
from scipy.io import wavfile  
import matplotlib.pyplot as plt  
import urllib.request  
import numpy as np  
import sys 

response = urllib.request.urlopen('http://www.thesoundarchive.com/austinpowers/smashingbaby.wav') 
print(response.info() )
WAV_FILE = 'smashingbaby.wav' 
filehandle = open(WAV_FILE, 'w') 
with open(WAV_FILE, 'wb') as filehandle:
    filehandle.write(response.read())  # 直接写入字节数据

# 可以选择读取并处理音频文件，或者做其他操作
print(f"{WAV_FILE} downloaded successfully!")
filehandle.close() 
sample_rate, data = wavfile.read(WAV_FILE)
print("Data type", data.dtype, "Shape", data.shape )

plt.subplot(2, 1, 1)  
plt.title("Original" ) 
plt.plot(data)  
plt.subplot(2, 1, 2) 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214140335391.png" alt="image-20250214140335391" style="zoom:50%;" />

`wavfile.read`该函数有两个返回值——采样率和音频数据。在本节教程中，我们只需要用到音频数据。

(2) 应用tile函数：

```python
# 重复音频片段 
# repeated = np.tile(data, int(sys.argv[1])) 
# 重复音频片段 
repeated = np.tile(data, 3) 
```

(3) 使用write函数写入一个新文件：

```python
plt.title("Repeated")  
plt.plot(repeated) 
wavfile.write("repeated_yababy.wav", sample_rate, repeated) 
plt.show () 
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214140519502.png" alt="image-20250214140519502" style="zoom:50%;" />

再次调整图片间距(在`plt.show()`之前)：

```python
# 调整子图之间的间距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.5)
```

下面是最终效果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214140630100.png" alt="image-20250214140630100" style="zoom:50%;" />

同时也有这两段wav:

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250214140655235.png" alt="image-20250214140655235" style="zoom:50%;" />

刚才做了些什么 :  我们读入了一个音频片段，将其重复三遍并将新数组写入了一个新的WAV文件。

# 10.21 本章小结

在本章中，我们只是触及了SciPy和SciKits的皮毛，学习了一点关于文件输入/输出、统计、信号处理、数学优化、插值以及音频和图像处理的知识。

在下一章中，我们将使用Pygame制作一些简单但有趣的游戏。Pygame是一个开源的Python游戏库。在这个过程中，我们将学习NumPy和Pygame的集成、SciKits机器学习模块以及其他内容。 