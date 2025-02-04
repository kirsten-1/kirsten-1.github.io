---
layout: post
title: "numpy(2)_numpy基础"
subtitle: "第 2 章 NumPy基础"
date: 2025-02-04
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 人工智能AI基础
---


[numpy(1)入门](https://kirsten-1.github.io/2025/01/27/numpy(1)_%E5%85%A5%E9%97%A8/)：介绍numpy的安装和numpy较纯python语言的简洁和性能优越性。

这篇主要介绍numpy基础。涵盖以下内容：

- 数据类型；
- 数组类型；
- 类型转换；
- 创建数组；
- 数组索引；
- 数组切片；
- 改变维度。

> 首先学习前，启动环境：
>
> ```shell
> workon env1
> jupyter notebook
> ```



# 2.1 NumPy 数组对象

NumPy中的`ndarray`是一个多维数组对象，该对象由两部分组成：

- 实际的数据；
- 描述这些数据的元数据。

> 其实元数据不是什么难懂的名词，就是用来**描述和解释数据的额外信息**。它不直接包含数据本身，而是描述数据的 **结构**、**类型**、**大小** 等。
>
> 例如：
>
> - **数据类型（dtype）**：描述数据的类型，如整数（int）、浮点数（float）等。
> - **形状（shape）**：描述数组的维度和每个维度的大小。比如一个二维数组 `3x2`（3行2列）或者三维数组。
> - **维度（ndim）**：描述数组的维度数量（1D、2D、3D等）。
> - **大小（size）**：数组中总共有多少个元素。
> - **步长（strides）**：描述每个维度中相邻元素的字节偏移量。
>
> 注：`ndarray`中，大致来说，n是指n个，d是dimension（维度）的意思。
>
> 可以把 **元数据** 看作是关于数据的 **说明书**，它告诉我们如何理解和操作这些数据，而 **实际的数据** 就是我们真正关心的内容，像数字或字符等。

**大部分的数组操作仅仅修改元数据部分，而不改变底层的实际数据。**

首先需大致了解以下内容：

1.在[numpy(1)入门](https://kirsten-1.github.io/2025/01/27/numpy(1)_%E5%85%A5%E9%97%A8/)中，已经知道如何使用arange函数创建数组。实际上，当时创建的数组只是包含一组数字的一维数组，而ndarray支持更高的维度。

例如：

```python
arr_3d = np.arange(12).reshape(2,3,2)
arr_3d
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250127230511933.png" alt="image-20250127230511933" style="zoom:50%;" />

2.NumPy数组一般是同质的（但有一种特殊的数组类型例外，它是异质的），即**数组中的所有元素类型必须是一致的**。这样有一个好处：如果知道数组中的元素均为同一类型，该数组所需的存储空间就很容易确定下来。

例如：

```python
arr_mixed = np.array([1,3.6,"9"])
arr_mixed.dtype
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250127230707338.png" alt="image-20250127230707338" style="zoom:50%;" />

> 类型混用时，也有优先级：str>float>int。所以上面最终的数组对象的每个元素的类型是str。

3.与Python中一样，NumPy数组的下标也是从0开始的。数组元素的数据类型用专门的对象表示。

4.数组的shape属性返回一个元组（tuple），元组中的元素即为NumPy数组每一个维度上的大小。

例如：

```python
arr_3d = np.arange(12).reshape(2,3,2)
arr_3d.shape
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250127230946128.png" alt="image-20250127230946128" style="zoom:50%;" />

# 2.2 动手实践：创建多维数组

> **面试题**：创建`ndarray`有哪些基本的方法？
>
> 1)**使用np.array()由python list创建**
>
> ```python
> list1 = [1, 2, 3, 5] 
> n = np.array(list1)
> ```
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203190515734.png" alt="image-20250203190515734" style="zoom:50%;" />
>
> ```python
> # 由ndarray变回list
> list2 = n.tolist()
> list2
> ```
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203190622683.png" alt="image-20250203190622683" style="zoom:50%;" />
>
> 注意：
>
> \- numpy默认ndarray的所有元素的类型是相同的(即**同质**)
>
> \- 如果传进来的列表中包含不同的类型，则统一为同一类型，优先级：`str>float>int`
>
> 例如：
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203190841141.png" alt="image-20250203190841141" style="zoom:50%;" />
>
> **2)使用np的routines函数创建**—-需要记住11个函数
>
> ```python
> np.ones(shape, dtype=None, order='C')
> np.zeros(shape, dtype=float, order='C')
> np.full(shape, fill_value, dtype=None, order='C')
> np.eye(N, M=None, k=0, dtype=float)      对角线为1其他的位置为0
> np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
> np.arange([start, ]stop, [step, ]dtype=None)
> np.random.randint(low, high=None, size=None, dtype='l')
> np.random.randn(d0, d1, ..., dn)  
> np.random.normal(loc=0.0, scale=1.0, size=None)
> np.random.random(size=None)  
> np.random.rand(d0, d1, d2, d3...,dn)
> ```
>
> 面试记忆方法：
>
> **一零满**：`np.ones`, `np.zeros`, `np.full`（都是填充的）
>
> **角线均**：`np.eye`, `np.linspace`, `np.arange`（对角线为1，均匀间隔，区间步长）
>
> **步长整**：`np.random.randint`（整数）
>
> **正态浮**：`np.random.randn`, `np.random.normal`, `np.random.random`, `np.random.rand`（正态分布和随机浮点数）
>
> ----
>
> 下面来看具体用法：
>
>
>
> 下面是对这些NumPy方法的详细介绍，包括它们的用法和需要注意的点：
>
> ------
>
> ### 1. **`np.ones(shape, dtype=None, order='C')`**
>
> - **功能**：生成一个由1填充的数组。
> - 参数：
    >   - `shape`：数组的形状，可以是整数或元组。比如，`(3, 4)`表示一个3行4列的矩阵。
>   - `dtype`：数据类型，默认为`float`。可以指定为任何NumPy支持的类型，如`int`，`float64`等。
>   - `order`：决定数组的内存布局。**`'C'`表示按行优先（C风格），`'F'`表示按列优先（Fortran风格）**。
> - **注意**：**如果不指定`dtype`，默认会生成浮点类型的数组**。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203192040149.png" alt="image-20250203192040149" style="zoom:50%;" />
>
> ### 2. **`np.zeros(shape, dtype=float, order='C')`**
>
> - **功能**：生成一个由0填充的数组。
> - **参数**：和`np.ones()`类似。
> - **注意**：常用于初始化一个空数组或矩阵。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203192148340.png" alt="image-20250203192148340" style="zoom:50%;" />
>
> ### 3. **`np.full(shape, fill_value, dtype=None, order='C')`**
>
> - **功能**：生成一个指定形状并填充指定值的数组。
> - 参数：
    >   - `shape`：数组的形状。
>   - `fill_value`：数组中每个元素的值。
>   - `dtype`：数据类型，默认为`None`，将根据`fill_value`自动推导。
>   - `order`：内存布局，默认为`'C'`。
> - **注意**：可以用来生成任何值的矩阵，不仅仅是0或1。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203192316428.png" alt="image-20250203192316428" style="zoom:50%;" />
>
> ### 4.**`np.eye(N, M=None, k=0, dtype=float)`**
>
> - **功能**：生成一个单位矩阵（对角线为1，其他位置为0）。
> - 参数：
    >   - `N`：矩阵的行数。
>   - `M`：矩阵的列数。如果不指定，默认为`N`，生成一个方阵。
>   - `k`：对角线的位置，`k=0`是主对角线，`k>0`是上对角线，`k<0`是下对角线。
>   - `dtype`：数据类型。
> - **注意**：如果需要生成其他对角线矩阵（如上对角线或下对角线），可以调整`k`的值。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203192505624.png" alt="image-20250203192505624" style="zoom:50%;" />
>
> ### 5. **`np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`**
>
> - **功能**：生成一个**指定范围内的均匀间隔的数字序列**。
> - 参数：
    >   - `start`：序列的起始值。
>   - `stop`：序列的结束值。
>   - `num`：生成的数字个数，默认是50。
>   - `endpoint`：是否包含`stop`值，**默认是`True`**。
>   - `retstep`：如果为`True`，返回间隔的大小，默认为`False`。
>   - `dtype`：数据类型。
> - **注意**：如果`endpoint=False`，序列会停止在`stop`之前，确保不会包含`stop`值。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250203193413514.png" alt="image-20250203193413514" style="zoom:50%;" />
>
> ### 6. **`np.arange([start, ]stop, [step, ]dtype=None)`**
>
> - **功能**：生成一个指定范围内的数字序列，**类似于Python的`range()`函数**。
> - 参数：
    >   - `start`：序列的起始值，默认为0。（可选）
>   - `stop`：序列的结束值（不包含）。（必须提供）
>   - `step`：步长，默认为1。（可选）
>   - `dtype`：数据类型。
> - **注意**：`stop`不包括在内。如果`step`为负数，`start`必须大于`stop`。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204133855678.png" alt="image-20250204133855678" style="zoom:50%;" />
>
> ### 7. **`np.random.randint(low, high=None, size=None, dtype='l')`**
>
> 生成指定范围内的随机整数。
>
> - 参数：
    >   - `low`：随机整数的下界。
>   - `high`：随机整数的上界。
>   - `size`：输出的数组形状。
>   - `dtype`：数据类型，默认为整数类型。
> - **注意**：返回的数组包含的是指定区间内的整数，且 **`high` 不包含在内**。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204134100416.png" alt="image-20250204134100416" style="zoom:50%;" />
>
> ### 8. **`np.random.randn(d0, d1, ..., dn)`**
>
> 生成**标准正态分布**的随机数（均值为 0，标准差为 1）。
>
> - 参数：
    >   - `d0, d1, ..., dn`：各维度的大小。
> - **注意**：返回的是一个服从标准正态分布的数组，形状由参数指定。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204134242605.png" alt="image-20250204134242605" style="zoom:50%;" />
>
> ### 9. **`np.random.normal(loc=0.0, scale=1.0, size=None)`**
>
> 生成符合**正态分布**的随机数，可以自定义均值和标准差。
>
> - 参数：
    >   - `loc`：均值，默认为 0。
>   - `scale`：标准差，默认为 1。
>   - `size`：输出的数组形状。
> - **注意**：生成的是符合指定均值和标准差的随机数。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204134404745.png" alt="image-20250204134404745" style="zoom:50%;" />
>
> ### 10. **`np.random.random(size=None)`**
>
> 生成 **0 到 1 之间的随机浮点数**。
>
> - 参数：
    >   - `size`：输出的数组形状。
> - **注意**：返回的是一个包含 0 到 1 之间的随机浮点数的数组。
>
> ### 11. **`np.random.rand(d0, d1, ..., dn)`**
>
> 生成均匀分布在 [0, 1) 区间的随机数。
>
> - 参数：
    >   - `d0, d1, ..., dn`：各维度的大小。
> - **注意**：类似于 `np.random.random()`，但接受的是维度参数而不是 `size`。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204134654618.png" alt="image-20250204134654618" style="zoom:50%;" />

既然我们已经知道如何创建向量，现在可以试着创建多维的NumPy数组，并查看其维度了。

(1) 创建一个多维数组。
(2) 显示该数组的维度。

```python
n = np.array([np.arange(2), np.arange(2)])
n
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204135210110.png" alt="image-20250204135210110" style="zoom:50%;" />

```python
n.shape
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204135227792.png" alt="image-20250204135227792" style="zoom:50%;" />

> 解释：`np.arange(2)` 会生成一个从 0 开始，到 2 之前的整数数组，即 `[0, 1]`。`[np.arange(2), np.arange(2)]` 创建了一个包含两个 `np.arange(2)` 生成的数组的列表，即`[[0, 1], [0, 1]]`。`np.array()` 将这个包含两个数组的列表转换为一个 NumPy 数组。最终的 `n` 数组是一个 2x2 的二维数组，其中每一行都是 `[0, 1]`。
>
> 将arange函数创建的数组作为列表元素，把这个列表作为参数传给array函数，从而创建了一个2×2的数组.

🌟**关于`np.array()`函数：**

**`array`函数可以依据给定的对象生成数组。给定的对象应是类数组，如Python中的列表。在上面的例子中，我们传给array函数的对象是一个NumPy数组的列表。像这样的类数组对象是array函数的唯一必要参数，其余的诸多参数均为有默认值的可选参数。**

----

**突击测验：ndarray对象维度属性的存储方式**
问题1 ndarray对象的维度属性是以下列哪种方式存储的？

(1) 逗号隔开的字符串  
(2) Python列表（list）
(3) Python元组（tuple）

> 答案：（3），`ndarray` 对象的维度属性是通过 **Python元组（tuple）** 来存储的。
>
> 在 NumPy 中，`ndarray` 的维度信息存储在 `.shape` 属性中，`shape` 是一个表示数组各维度大小的 **元组（tuple）**。
>
> - **例如**：对于一个 3x2 的数组，`shape` 属性会是 `(3, 2)`，这意味着数组有 3 行 2 列。
> - 如果是一个 3x2x4 的三维数组，`shape` 会是 `(3, 2, 4)`。
>
> ### 为什么是元组（tuple）：
>
> - 元组是一个**不可变**的序列，符合存储数组维度这一信息的需求，因为数组的维度一旦确定，就不需要修改它。
> - 与列表（list）不同，元组的不可变性使得它在存储维度信息时更加合适和高效。

----

勇敢出发：创建3×3的多维数组
现在，创建一个3×3的多维数组应该不是一件难事。试试看，并在创建多维数组后检查其维度是否与你设想的一致。

方法1:

```python
n1 = np.array([np.arange(3), np.arange(3), np.arange(3)])
n1
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204140344731.png" alt="image-20250204140344731" style="zoom:50%;" />

方法2:

```python
n2 = np.random.randint(0, 3, (3, 3))
n2
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204140409673.png" alt="image-20250204140409673" style="zoom:50%;" />

通过 `.shape` 确保数组的维度为 `(3, 3)`，即所设想的 3×3 维度。

方法3:

```python
n3 = np.ones((3, 3))
n3
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204140626114.png" alt="image-20250204140626114" style="zoom:50%;" />

方法4:

```python
n4 = np.zeros((3, 3))
n4
```



<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204140652838.png" alt="image-20250204140652838" style="zoom:50%;" />

方法5:

```python
n5 = np.full((3, 3), 5)
n5
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204140805225.png" alt="image-20250204140805225" style="zoom:50%;" />

方法6:

```python
n6 = np.eye(3)
n6
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204140931851.png" alt="image-20250204140931851" style="zoom:50%;" />

方法7:

```python
n7 = np.random.randn(3,3)
n7
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141035692.png" alt="image-20250204141035692" style="zoom:50%;" />

方法8:

```python
n8 = np.random.normal(size = (3, 3))
n8
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141131670.png" alt="image-20250204141131670" style="zoom:50%;" />

方法9:

```python
n9 = np.random.random((3, 3))
n9
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141217451.png" alt="image-20250204141217451" style="zoom:50%;" />

方法10:

```python
n10 = np.random.rand(3, 3)
n10
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141253734.png" alt="image-20250204141253734" style="zoom:50%;" />

方法11:

```python
n11 = np.linspace(0, 10, num = 9).reshape(3,3)  
n11
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141450851.png" alt="image-20250204141450851" style="zoom:50%;" />

方法12:

```python
n12 = np.arange(9).reshape(3, 3)
n12
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141555578.png" alt="image-20250204141555578" style="zoom:50%;" />

----

另外拓展一些方法：

方法13:`np.random.choice()` 可以从一个给定的一维数组中随机选择元素，生成一个指定形状的数组。

```python
n13 = np.random.choice([1, 2, 3], size = (3, 3))
n13
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141822155.png" alt="image-20250204141822155" style="zoom:50%;" />

方法14:`np.empty()` 创建一个未初始化的数组，**内存中的值是随机的**。

```python
n14 = np.empty((3, 3))
n14
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204141921125.png" alt="image-20250204141921125" style="zoom:50%;" />

还有其他方法，不再列举。

----

## 2.2.1 选取数组元素

有时候，我们需要选取数组中的某个特定元素。首先还是创建一个2×2的多维数组：

```python
n = np.array([[1, 2], [3, 4]])
n
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204142203842.png" alt="image-20250204142203842" style="zoom:50%;" />

在创建这个多维数组时，我们给array函数传递的对象是一个嵌套的列表。现在来依次选取
该数组中的元素。记住，**数组的下标是从0开始的**。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204142336593.png" alt="image-20250204142336593" style="zoom:50%;" />

从数组中选取元素就是这么简单。**对于数组a，只需要用a[m,n]选取各数组元素，其中m和n为元素下标，对应的位置如下表所示**。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204142414321.png" alt="image-20250204142414321" style="zoom:50%;" />

## 2.2.2 NumPy 数据类型

> 回顾面试题：python的数据类型有哪些？
>
> | 类型         | 描述                                     | 示例                        |
> | ------------ | ---------------------------------------- | --------------------------- |
> | `int`        | 整数类型                                 | `x = 42`                    |
> | `float`      | 浮点数类型                               | `y = 3.14`                  |
> | `complex`    | 复数类型                                 | `z = 1 + 2j`                |
> | `str`        | 字符串类型，表示文本                     | `name = "Alice"`            |
> | `list`       | 列表类型，表示可变有序集合               | `numbers = [1, 2, 3]`       |
> | `tuple`      | 元组类型，表示不可变有序集合             | `coords = (1, 2, 3)`        |
> | `range`      | 范围类型，表示数字的不可变序列           | `r = range(0, 10)`          |
> | `set`        | 集合类型，表示无序不重复元素集合         | `s = {1, 2, 3}`             |
> | `frozenset`  | 冻结集合类型，表示不可变集合             | `fs = frozenset([1, 2, 3])` |
> | `dict`       | 字典类型，表示键值对集合                 | `d = {"key": "value"}`      |
> | `bool`       | 布尔类型，表示真 (`True`) 或假 (`False`) | `flag = True`               |
> | `bytes`      | 不可变字节序列                           | `b = b"hello"`              |
> | `bytearray`  | 可变字节序列                             | `ba = bytearray([65, 66])`  |
> | `memoryview` | 内存视图，用于访问对象的缓冲区协议       | `m = memoryview(b"hello")`  |
> | `NoneType`   | `None`，表示“无”或“空值”                 | `n = None`                  |

Python支持的数据类型有整型、浮点型以及复数型，但这些类型不足以满足科学计算的需求，因此NumPy添加了很多其他的数据类型。在实际应用中，我们需要不同精度的数据类型，它们占用的内存空间也是不同的。在NumPy中，大部分数据类型名是以数字结尾的，这个数字表示其在内存中占用的位数。下面的表格（整理自[NumPy用户手册-NumPy基础知识-数据类型](https://numpy.com.cn/doc/stable/user/basics.types.html)）列出了NumPy中支持的数据类型。

| 类型         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| `bool`       | 用一位存储的布尔类型（值为 TRUE 或 FALSE）                   |
| `inti`       | 由所在平台决定其精度的整数（一通常为 int32 或 int64）        |
| `int8`       | 整数，范围从 -128 到 127                                     |
| `int16`      | 整数，范围从 -32,768 到 32,767                               |
| `int32`      | 整数，范围从 -2³¹ 到 2³¹-1                                   |
| `int64`      | 整数，范围从 -2⁶³ 到 2⁶³-1                                   |
| `uint8`      | 无符号整数，范围从 0 到 255                                  |
| `uint16`     | 无符号整数，范围从 0 到 65,535                               |
| `uint32`     | 无符号整数，范围从 0 到 2³²-1                                |
| `uint64`     | 无符号整数，范围从 0 到 2⁶⁴-1                                |
| `float16`    | 半精度浮点数（16 位）：其中用 1 位表示正负号，5 位表示指数，10 位表示尾数 |
| `float32`    | 单精度浮点数（32 位）：其中用 1 位表示正负号，8 位表示指数，23 位表示尾数 |
| `float64`    | 双精度浮点数（64 位）：其中用 1 位表示正负号，11 位表示指数，52 位表示尾数 |
| `complex64`  | 复数，分别用两个 32 位浮点数表示实部和虚部                   |
| `complex128` | 复数，分别用两个 64 位浮点数表示实部和虚部                   |

每一种数据类型均有对应的类型转换函数：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204143651160.png" alt="image-20250204143651160" style="zoom:50%;" />

在NumPy中，许多函数的参数中可以指定数据类型，通常这个参数是可选的：

```python
n = np.arange(7, dtype = "float64")
n
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204143750905.png" alt="image-20250204143750905" style="zoom:50%;" />

需要注意的是，复数是不能转换为整数的，这将触发TypeError错误： `TypeError: can't convert complex to int`

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204143853551.png" alt="image-20250204143853551" style="zoom:50%;" />

同样，复数也不能转换为浮点数。不过，浮点数却可以转换为复数，例如complex(1.0)。注意，有j的部分为复数的虚部。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204143946243.png" alt="image-20250204143946243" style="zoom:50%;" />

## 2.2.3 数据类型对象

**数据类型对象是numpy.dtype类的实例**。如前所述，NumPy数组是有数据类型的，更确切地说，NumPy数组中的每一个元素均为相同的数据类型(“同质”)。数据类型对象可以给出单个数组元素在内存中占用的字节数，即dtype类的itemsize属性：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204144143606.png" alt="image-20250204144143606" style="zoom:50%;" />

> `dtype` 类的 `itemsize` 属性用于返回每个元素的字节大小（单位是字节）。`itemsize` 是一个非常有用的属性，可以帮助我们了解一个数组中每个元素的内存占用情况。
>
> 例如，如果数据类型是 `int32`，每个元素占用 4 个字节；如果是 `float64`，每个元素占用 8 个字节。
>
> 如果你需要知道数组中所有元素的总内存占用，可以使用 `.nbytes`（它会返回整个数组占用的字节总数），而 `itemsize` 是返回单个元素的字节数。
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204144323122.png" alt="image-20250204144323122" style="zoom:50%;" />

## 2.2.4 字符编码 (也叫字符代码)

NumPy可以使用字符编码来表示数据类型，这是为了兼容NumPy的前身Numeric。我不推荐使用字符编码，但有时会用到，因此下面还是列出了字符编码的对应表。读者应该优先使用dtype 对象来表示数据类型，而不是这些字符编码。

| 数据类型       | 字符编码 |
| -------------- | -------- |
| 整数           | i        |
| 无符号整数     | u        |
| 单精度浮点数   | f        |
| 双精度浮点数   | d        |
| 布尔值         | b        |
| 复数           | D        |
| 字符串         | S        |
| unicode 字符串 | U        |
| void（空）     | V        |

下面的代码创建了一个单精度浮点数数组：

```python
f = np.arange(8, dtype = "f")
f
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204144647056.png" alt="image-20250204144647056" style="zoom:50%;" />

与此类似，还可以创建一个复数数组：

```python
com = np.arange(8, dtype = "D")
com
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204144757573.png" alt="image-20250204144757573" style="zoom:50%;" />

## 2.2.5 自定义数据类型

我们有很多种自定义数据类型的方法，以浮点型为例。

- **使用 Python 自带的浮点数类型**：

    - ```python
    np.dtype(float)
    ```

    - `dtype(float)`：返回 `dtype('float64')`，表示使用 Python 的标准浮点数类型，这通常对应于 64 位浮点数。

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204145333108.png" alt="image-20250204145333108" style="zoom:50%;" />

- **使用字符编码指定单精度浮点数类型**：

    - ```python
    np.dtype("f")
    ```

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204145403402.png" alt="image-20250204145403402" style="zoom:50%;" />

    - `dtype('f')`：返回 `dtype('float32')`，表示创建一个 32 位单精度浮点数类型。

- **使用字符编码指定双精度浮点数类型**：

    - ```python
    np.dtype("d")
    ```

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204145448332.png" alt="image-20250204145448332" style="zoom:50%;" />

    - `dtype('d')`：返回 `dtype('float64')`，表示创建一个 64 位双精度浮点数类型。

- **使用两个字符作为参数**：

    - 将两个字符作为参数传给数据类型的构造函数。此时，第一个字符表示数据类型，
      第二个字符表示该类型在内存中占用的字节数（2、4、8分别代表精度为16、32、64位的浮点数）：

    - 第一个字符表示数据类型（如 `f`、`d` 等），第二个字符表示该类型的精度，即内存中占用的字节数。

        - `2` 表示 16 位精度的浮点数
        - `4` 表示 32 位精度的浮点数
        - `8` 表示 64 位精度的浮点数

    - ```python
    np.dtype("f8")
    ```

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204145651525.png" alt="image-20250204145651525" style="zoom:50%;" />



完整的NumPy数据类型列表可以在`sctypeDict.keys()`中找到：

```python
np.sctypeDict.keys()
输出：
dict_keys(['?', 0, 'byte', 'b', 1, 'ubyte', 'B', 2, 'short', 'h', 3, 'ushort', 'H', 4, 'i', 5, 'uint', 'I', 6, 'intp', 'p', 7, 'uintp', 'P', 8, 'long', 'l', 'ulong', 'L', 'longlong', 'q', 9, 'ulonglong', 'Q', 10, 'half', 'e', 23, 'f', 11, 'double', 'd', 12, 'longdouble', 'g', 13, 'cfloat', 'F', 14, 'cdouble', 'D', 15, 'clongdouble', 'G', 16, 'O', 17, 'S', 18, 'unicode', 'U', 19, 'void', 'V', 20, 'M', 21, 'm', 22, 'b1', 'bool8', 'i8', 'int64', 'u8', 'uint64', 'f2', 'float16', 'f4', 'float32', 'f8', 'float64', 'f16', 'float128', 'c8', 'complex64', 'c16', 'complex128', 'c32', 'complex256', 'object0', 'bytes0', 'str0', 'void0', 'M8', 'datetime64', 'm8', 'timedelta64', 'int32', 'i4', 'uint32', 'u4', 'int16', 'i2', 'uint16', 'u2', 'int8', 'i1', 'uint8', 'u1', 'complex_', 'single', 'csingle', 'singlecomplex', 'float_', 'intc', 'uintc', 'int_', 'longfloat', 'clongfloat', 'longcomplex', 'bool_', 'bytes_', 'string_', 'str_', 'unicode_', 'object_', 'int', 'float', 'complex', 'bool', 'object', 'str', 'bytes', 'a', 'int0', 'uint0'])
```

## 2.2.6 dtype 类的属性

dtype类有很多有用的属性。例如，我们可以获取数据类型的字符编码：

```
a = np.dtype(float)
a
a.char
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204145925888.png" alt="image-20250204145925888" style="zoom:50%;" />

type属性对应于数组元素的数据类型：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204150014622.png" alt="image-20250204150014622" style="zoom:50%;" />

str属性可以给出数据类型的字符串表示，该字符串的首个字符表示**字节序（endianness）**，
后面如果还有字符的话，将是一个字符编码，接着一个数字表示每个数组元素存储所需的字节数。这里，字节序是指位长为32或64的字（word）存储的顺序，包括大端序（big-endian）和小端序（little-endian）。大端序是将最高位字节存储在最低的内存地址处，用>表示；与之相反，小端序是将最低位字节存储在最低的内存地址处，用<表示：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204150124650.png" alt="image-20250204150124650" style="zoom:50%;" />

----

# 2.3 动手实践：创建自定义数据类型

自定义数据类型是一种**异构**数据类型，可以当做用来记录电子表格或数据库中一行数据的结构。作为示例，我们将创建一个存储商店库存信息的数据类型。其中，我们用一个长度为40个字符的字符串来记录商品名称，用一个32位的整数来记录商品的库存数量，最后用一个32位的单精度浮点数来记录商品价格。下面是具体的步骤。

(1) 创建数据类型：

```python
t = np.dtype([('name', np.str_, 40), ('numitems', np.int32), ('price', np.float32)])
t
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204153450783.png" alt="image-20250204153450783" style="zoom:50%;" />

> 这段代码定义了一个复合数据类型（structured dtype）`t`，它描述了一个包含多个字段的数据结构。具体来说，这个 `dtype` 用于定义一个包含三列的结构体，其中每个字段具有不同的数据类型和不同的长度。`np.dtype()` 是 NumPy 中的函数，用于创建自定义的数据类型。其中的参数是一个包含字段名称和数据类型的列表，用来定义复合数据类型（也叫结构化数据类型）。

(2) 查看数据类型（也可以查看某一字段的数据类型） :

```python
t["name"]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204153646947.png" alt="image-20250204153646947" style="zoom:50%;" />

在用array函数创建数组时，如果没有在参数中指定数据类型，将默认为浮点数类型。而现在，我们想要创建自定义数据类型的数组，就必须在参数中指定数据类型，否则将触发TypeError 错误：

```python
items = np.array([('Meaning of life DVD', 42, .1), ("butter", 56, 9)], dtype = t)
items
items[1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204153926826.png" alt="image-20250204153926826" style="zoom:50%;" />

刚才做了些什么 :我们创建了一种自定义的异构数据类型，该数据类型包括一个用字符串记录的名字、一个用整数记录的数字以及一个用浮点数记录的价格。

---

# 2.4 一维数组的索引和切片

一维数组的切片操作与Python列表的切片操作很相似。例如，我们可以用下标`3~7`来选取元素`3~6`：

```python
a = np.arange(9)
a
a[3:7]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154117756.png" alt="image-20250204154117756" style="zoom:50%;" />

也可以用下标0~7，以2为步长选取元素：

```python
a[:7:2]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154153573.png" alt="image-20250204154153573" style="zoom:50%;" />

和Python中一样，我们也可以利用负数下标翻转数组：

```python
a[::-1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154225047.png" alt="image-20250204154225047" style="zoom:50%;" />

# 2.5 动手实践：多维数组的切片和索引

ndarray支持在多维数组上的切片操作。为了方便起见，我们可以用一个省略号`（...）`来表示遍历剩下的维度。

(1) 举例来说，我们先用arange函数创建一个数组并改变其维度，使之变成一个三维数组：

```python
n = np.arange(24).reshape(2, 3, 4)
n
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154416354.png" alt="image-20250204154416354" style="zoom:50%;" />

多维数组n中有0~23的整数，共24个元素，是一个`2×3×4`的三维数组。我们可以形象地把它看做一个两层楼建筑，每层楼有12个房间，并排列成3行4列。或者，我们也可以将其看成是电子表格中工作表（sheet）、行和列的关系。你可能已经猜到，**reshape函数的作用是改变数组的“形状”，也就是改变数组的维度，其参数为一个正整数元组，分别指定数组在每个维度上的大小。如果指定的维度和数组的元素数目不相吻合，函数将抛出异常。** (reshape函数要求前后数组的总个数是相同的)

(2) 我们可以用三维坐标来选定任意一个房间，即楼层、行号和列号。例如，选定第1层楼、第1行、第1列的房间（也可以说是第0层楼、第0行、第0列，这只是习惯问题），可以这样表示：

```python
n[0, 0, 0]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154620589.png" alt="image-20250204154620589" style="zoom:50%;" />

(3) 如果我们不关心楼层，也就是说要选取所有楼层的第1行、第1列的房间，那么可以将第1个下标用英文标点的冒号`:`来代替：

```python
n[:, 0, 0]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154803434.png" alt="image-20250204154803434" style="zoom:50%;" />

我们还可以这样写，选取第1层楼的所有房间：

```python
n[0]
```

或者

```python
n[0, :, :]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154910336.png" alt="image-20250204154910336" style="zoom:50%;" />

多个冒号可以用一个省略号`（...）`来代替，因此上面的代码等价于：

```python
n[0, ...]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204154944115.png" alt="image-20250204154944115" style="zoom:50%;" />

进而可以选取第1层楼、第2排的所有房间：

```python
n[0, 1, :]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155031773.png" alt="image-20250204155031773" style="zoom:50%;" />

或者：

```python
n[0, 1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155102412.png" alt="image-20250204155102412" style="zoom:50%;" />

(4) 再进一步，我们可以在上面的数组切片中间隔地选定元素：

```python
n[0, 1, ::2]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155143156.png" alt="image-20250204155143156" style="zoom:50%;" />

(5) 如果要选取所有楼层的位于第2列的房间，即不指定楼层和行号，用如下代码即可：

```python
n[:, :, 1]
或者
n[..., 1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155327020.png" alt="image-20250204155327020" style="zoom:50%;" />

类似地，我们可以选取所有位于第2行的房间，而不指定楼层和列号：

```python
n[:, 1, :]
或者
n[:, 1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155440611.png" alt="image-20250204155440611" style="zoom:50%;" />

如果要选取第1层楼的所有位于第2列的房间，在对应的两个维度上指定即可：

```python
n[0, :, 1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155544201.png" alt="image-20250204155544201" style="zoom:50%;" />

(6) 如果要选取第1层楼的最后一列的所有房间，使用如下代码：

```python
n[0, :, 3]
或者
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155825259.png" alt="image-20250204155825259" style="zoom:50%;" />

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160048177.png" alt="image-20250204160048177" style="zoom:50%;" />

如果要反向选取第1层楼的最后一列的所有房间，使用如下代码：

```python
n[0, ::-1, -1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204155952339.png" alt="image-20250204155952339" style="zoom:50%;" />

在该数组切片中间隔地选定元素：

```python
n[0, ::2, -1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160202755.png" alt="image-20250204160202755" style="zoom:50%;" />

如果在多维数组中执行翻转一维数组的命令，将在最前面的维度上翻转元素的顺序，在我们的例子中将把第1层楼和第2层楼的房间交换：

```python
n[::-1]
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160318908.png" alt="image-20250204160318908" style="zoom:50%;" />

刚才做了些什么 :我们用各种方法对一个NumPy多维数组进行了切片操作。

# 2.6 动手实践：改变数组的维度

我们已经学习了怎样使用reshape函数，现在来学习一下怎样将数组展平。

**(1) ravel 我们可以用ravel函数完成展平的操作：**

```python
n = np.arange(12).reshape(3,4)
n
n.ravel()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160449792.png" alt="image-20250204160449792" style="zoom:50%;" />

**(2) flatten 这个函数恰如其名，flatten就是展平的意思，与ravel函数的功能相同。**
**不过，flatten函数会请求分配内存来保存结果，而ravel函数只是返回数组的一个视图（view）**：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160550982.png" alt="image-20250204160550982" style="zoom:50%;" />

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160639313.png" alt="image-20250204160639313" style="zoom:50%;" />

**(3) 用元组设置维度 除了可以使用reshape函数，我们也可以直接用一个正整数元组来设置数组的维度，如下所示：**

```python
n.shape = (2, 6)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204160724262.png" alt="image-20250204160724262" style="zoom:50%;" />

正如你所看到的，这样的做法**将直接改变所操作的数组**，现在数组n成了一个2×6的多维数组。

**(4) transpose 在线性代数中，转置矩阵是很常见的操作。对于多维数组，我们也可以这样做：**

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161217070.png" alt="image-20250204161217070" style="zoom:50%;" />

**(5) resize resize和reshape函数的功能一样，但resize会直接修改所操作的数组：**

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161332514.png" alt="image-20250204161332514" style="zoom:50%;" />

刚才做了些什么 :我们用ravel、flatten、reshape和resize函数对NumPy数组的维度进行了修改。

# 2.7 数组的组合

NumPy数组有水平组合、垂直组合和深度组合等多种组合方式，我们将使用vstack、dstack、hstack、column_stack、row_stack以及concatenate函数来完成数组的组合。

# 2.8 动手实践：组合数组

首先，我们来创建一些数组：

```python
a = np.arange(9).reshape(3, 3)
a
b = a*2
b
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161514566.png" alt="image-20250204161514566" style="zoom:50%;" />

**(1) 水平组合** 我们先从水平组合开始练习。将ndarray对象构成的元组作为参数，传给hstack函数。如下所示：

```python
np.hstack((a, b))
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161632412.png" alt="image-20250204161632412" style="zoom:50%;" />

我们也可以用concatenate函数来实现同样的效果，如下所示：

```python
np.concatenate((a, b), axis = 1)
```

![image-20250204161758588](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161758588.png)

总结如下图：



![image-20250204161822596](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161822596.png)

**(2) 垂直组合** 垂直组合同样需要构造一个元组作为参数，只不过这次的函数变成了vstack。如下所示：

```python
np.vstack((a,b))
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204161858860.png" alt="image-20250204161858860" style="zoom:50%;" />

同样，我们将concatenate函数的axis参数设置为0即可实现同样的效果。这也是axis参数的默认值：

```python
np.concatenate((a, b))
或者
np.concatenate((a, b), axis = 0)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162007821.png" alt="image-20250204162007821" style="zoom:50%;" />

总结如下：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162023399.png" alt="image-20250204162023399" style="zoom:50%;" />

**(3) 深度组合** 将相同的元组作为参数传给dstack函数，即可完成数组的深度组合。所谓

深度组合，就是将一系列数组沿着纵轴（深度）方向进行层叠组合。举个例子，有若干张二维平面内的图像点阵数据，我们可以将这些图像数据沿纵轴方向层叠在一起，这就形象地解释了什么是深度组合。

```python
np.dstack((a,b))
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162158012.png" alt="image-20250204162158012" style="zoom:50%;" />

**(4) 列组合** `column_stack`函数对于一维数组将按列方向进行组合，如下所示：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162334219.png" alt="image-20250204162334219" style="zoom:50%;" />

而对于二维数组，column_stack与hstack的效果是相同的：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162454275.png" alt="image-20250204162454275" style="zoom:50%;" />

是的，你猜对了！我们可以用==运算符来比较两个NumPy数组，是不是很简洁？

**(5) 行组合** 当然，NumPy中也有按行方向进行组合的函数，它就是row_stack。对于两 个一维数组，将直接层叠起来组合成一个二维数组。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162626890.png" alt="image-20250204162626890" style="zoom:50%;" />

对于二维数组，row_stack与vstack的效果是相同的：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204162827029.png" alt="image-20250204162827029" style="zoom:50%;" />

刚才做了些什么 :我们按照水平、垂直和深度等方式进行了组合数组的操作。我们使用了vstack、dstack、hstack、column_stack、row_stack以及concatenate函数。

-----

# 2.9 数组的分割

NumPy数组可以进行水平、垂直或深度分割，相关的函数有hsplit、vsplit、dsplit和split。我们可以将数组分割成相同大小的子数组，也可以指定原数组中需要分割的位置。

# 2.10 动手实践：分割数组

**(1) 水平分割** 下面的代码将把数组沿着水平方向分割为3个相同大小的子数组：

`np.hsplit(ary, indices_or_sections)`:

- **`ary`**：待分割的输入数组，通常是一个二维数组（矩阵）。

- **`indices_or_sections`**：

    - 如果是整数，则表示将数组平均分成 `indices_or_sections` 个子数组。

    - 如果是一个一维数组，则它应该包含沿着水平方向分割的索引位置（即要在哪些列处分割）。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204174139618.png" alt="image-20250204174139618" style="zoom:50%;" />

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204174354750.png" alt="image-20250204174354750" style="zoom:50%;" />

对同样的数组，调用split函数并在参数中指定参数axis=1，对比一下结果：

```python
np.split(n, 2, axis = 1)
```



<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204174451178.png" alt="image-20250204174451178" style="zoom:50%;" />

**(2) 垂直分割** vsplit函数将把数组沿着垂直方向分割：

```python
np.vsplit(n, 3)
```



<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204180530102.png" alt="image-20250204180530102" style="zoom:50%;" />

同样，调用split函数并在参数中指定参数`axis=0`，也可以得到同样的结果：

```python
np.split(n, 3, axis = 0)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204180707325.png" alt="image-20250204180707325" style="zoom:50%;" />

(3) 深度分割 不出所料，dsplit函数将按深度方向分割数组。我们先创建一个三维数组：

> `np.dsplit` 是 NumPy 中用于将一个三维数组沿着深度（即沿着第三个轴，通常是数组的“层”或“深度”）进行分割的函数。它的使用方式与 `np.hsplit`（水平分割）和 `np.vsplit`（垂直分割）类似，但是 `np.dsplit` 适用于三维数组，按深度（第三个轴）进行分割。
>
> `np.dsplit(ary, indices_or_sections)`
>
> - **`ary`**：待分割的三维数组。
>
> - **`indices_or_sections`**：
    >
    >   - 如果是整数，则表示将数组沿深度方向分割成 `indices_or_sections` 个子数组。
>
>   - 如果是一个一维数组，它应该包含沿深度方向的索引位置，表示在哪些“深度”进行分割。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181148193.png" alt="image-20250204181148193" style="zoom:50%;" />

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181210420.png" alt="image-20250204181210420" style="zoom:50%;" />

刚才做了些什么 :我们用hsplit、vsplit、dsplit和split函数进行了分割数组的操作。

# 2.11 数组的属性

除了shape和dtype属性以外，ndarray对象还有很多其他的属性，在下面一一列出。

- ndim属性，给出数组的维数，或数组轴的个数：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181329553.png" alt="image-20250204181329553" style="zoom:50%;" />
    - 简单理解，有几个`[`，`ndim`就是几。

- size属性，给出数组元素的总个数

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181421419.png" alt="image-20250204181421419" style="zoom:50%;" />

- itemsize属性，给出数组中的**元素**在内存中所占的字节数： (注意：问的是单个元素)

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181523133.png" alt="image-20250204181523133" style="zoom:50%;" />

- 如果你想知道整个数组所占的存储空间，可以用nbytes属性来查看。这个属性的值其实就是itemsize和size属性值的乘积：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181621611.png" alt="image-20250204181621611" style="zoom:50%;" />

- T属性的效果和transpose函数一样:

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181710040.png" alt="image-20250204181710040" style="zoom:50%;" />

- 对于一维数组，其T属性就是原数组：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181749313.png" alt="image-20250204181749313" style="zoom:50%;" />

- 在NumPy中，复数的虚部是用j表示的。例如，我们可以创建一个由复数构成的数组：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181848219.png" alt="image-20250204181848219" style="zoom:50%;" />

- real属性，给出复数数组的实部。如果数组中只包含实数元素，则其real属性将输出原数组：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181921453.png" alt="image-20250204181921453" style="zoom:50%;" />

- imag属性，给出复数数组的虚部：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204181954556.png" alt="image-20250204181954556" style="zoom:50%;" />

- 如果数组中包含复数元素，则其数据类型自动变为复数型：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182027932.png" alt="image-20250204182027932" style="zoom:50%;" />

- flat属性将返回一个numpy.flatiter对象，这是获得flatiter对象的唯一方式——我们无法访问flatiter的构造函数。这个所谓的“扁平迭代器”可以让我们像遍历一维数组一样去遍历任意的多维数组，如下所示：

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182133924.png" alt="image-20250204182133924" style="zoom:50%;" />

    - 补充：`numpy.flatiter` 是 NumPy 中的一个迭代器类，用于以扁平化（flattened）方式遍历数组的所有元素。它允许我们在数组中按顺序访问每个元素，无论原数组的维度如何。`flatiter` 是一个数组迭代器，它返回一个一维的迭代器，即使数组是多维的，它也会将数组展平并按元素顺序遍历。

        - `flatiter` 允许你通过单一的迭代接口遍历多维数组，而无需关心数组的实际维度。

    - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182336727.png" style="zoom:50%;" />

    - 我们还可以用flatiter对象直接获取一个数组元素：

        - ```python
      item[1]
      ```

        - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182443215.png" alt="image-20250204182443215" style="zoom:50%;" />

        - 或者`n.flat[1]`

    - 或者获取多个元素：

        - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182619739.png" alt="image-20250204182619739" style="zoom:50%;" />

    - flat属性是一个可赋值的属性。对flat属性赋值将导致整个数组的元素都被覆盖：

        - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182653632.png" alt="image-20250204182653632" style="zoom:50%;" />
        - <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182722751.png" alt="image-20250204182722751" style="zoom:50%;" />

----

所有学习过的`ndarray`属性，总结如下：

![image-20250204182754108](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182754108.png)

# 2.12 动手实践：数组的转换

我们可以使用tolist函数将NumPy数组转换成Python列表。

(1) 转换成列表：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182907399.png" alt="image-20250204182907399" style="zoom:50%;" />

(2) astype函数可以在转换数组时指定数据类型：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204182953688.png" alt="image-20250204182953688" style="zoom:50%;" />

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204183119477.png" alt="image-20250204183119477" style="zoom:50%;" />

在上面将复数转换为整数的过程中，我们丢失了复数的虚部。astype函数也可以接受数据类型为字符串的参数。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250204183202389.png" alt="image-20250204183202389" style="zoom:50%;" />

这一次我们使用了正确的数据类型，因此不会再显示任何警告信息。

刚才做了些什么 :我们将NumPy数组转换成了不同数据类型的Python列表。

# 2.13 本章小结

在本章中，我们学习了很多NumPy的基础知识：数据类型和NumPy数组。对于数组而言，有很多属性可以用来描述数组，数据类型就是其中之一。在NumPy中，数组的数据类型是用对象来完善表示的。

类似于Python列表，NumPy数组也可以方便地进行切片和索引操作。在多维数组上，NumPy有明显的优势。

涉及改变数组维度的操作有很多种——组合、调整、设置维度和分割等。在这一章中，对很多改变数组维度的实用函数进行了说明。

在学习完基础知识后，我们将进入到第3章来学习NumPy中的常用函数，包括基本数学函数和统计函数等。