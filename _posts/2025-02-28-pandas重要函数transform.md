---
layout: post
title: "pandas超级重要函数-transform函数"
subtitle: "返回与输入组具有相同形状的结果的transform函数"
date: 2025-02-28
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- pandas
---


<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>


[TOC]

NumPy 博客总结：

[《Python数据分析基础教程：NumPy学习指南（第2版）》所有章节阅读笔记+代码](https://kirsten-1.github.io/2025/02/14/NumPy%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97(%E7%AC%AC2%E7%89%88)%E9%98%85%E8%AF%BB%E6%80%BB%E7%BB%93/)

[70道NumPy 面试题(题目+答案)](https://kirsten-1.github.io/2025/02/21/NumPy70%E9%A2%98/)

pandas博客总结：

[pandas(1)数据预处理](https://kirsten-1.github.io/2025/02/21/Pandas(1)%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86/)

[pandas(2)数据分析](https://kirsten-1.github.io/2025/02/24/Pandas(2)%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90/)

[pandas(3)常用函数操作](https://kirsten-1.github.io/2025/02/24/pandas(3)%E5%B8%B8%E7%94%A8%E5%87%BD%E6%95%B0%E6%93%8D%E4%BD%9C/)

[pandas(4)大数据处理技巧](https://kirsten-1.github.io/2025/02/25/Pandas(4)%E5%A4%A7%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%8A%80%E5%B7%A7/)

[【力扣】pandas入门15题](https://kirsten-1.github.io/2025/02/25/%E5%8A%9B%E6%89%A3pandas%E5%85%A5%E9%97%A815%E9%A2%98/)

---

aggregation会返回数据的缩减版本，而transform能返回完整数据的某一变换版本供我们重组。这样的transformation，**输出的形状和输入一致。**

> 如果想要体会下transform函数的魅力，建议可以通过一道力扣的题目体会：[184. 部门工资最高的员工](https://leetcode.cn/problems/department-highest-salary/)

`transform() `函数的主要作用是在分组数据上进行逐元素的转换，并保持原始数据的索引结构。 它与 `apply() `函数类似，但` apply() `函数可以返回任意形状的结果，而` transform() `必须返回与输入组具有相同形状的结果。

函数签名：

```python
df.groupby(by)[column].transform(func, *args, **kwargs)
```

- **func**: 要应用的函数。 它可以是：
    - **内置函数**: 例如 sum, mean, max, min 等。 但通常不直接使用内置函数，而是结合 lambda 表达式或自定义函数使用。
    - **lambda 表达式**: 用于定义简单的匿名函数。
    - **自定义函数**: 可以定义更复杂的函数，进行更精细的转换。
- **`*args`**: 传递给 func 的位置参数。
- **`**kwargs`**: 传递给 func 的关键字参数。(`kw=keyword`)

---

# **标准化 (Standardization) 或归一化 (Normalization)**

将每个组的数据缩放到一个特定的范围，或者使其具有特定的均值和标准差。

```python
import pandas as pd

data = {'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
        'Value': [10, 15, 20, 25, 12, 22]}
df = pd.DataFrame(data)
display(df)
# 对每个 Category 组的 Value 列进行标准化 (Z-score)
df['Value_Standardized'] = df.groupby('Category')['Value'].transform(lambda x: (x - x.mean()) / x.std())
print(df)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250228132418866.png" alt="image-20250228132418866" style="zoom:50%;" />

# **缺失值填充 (Missing Value Imputation)**

使用 transform() 基于组的均值、中位数或其他统计量来填充缺失值,结合`fillna`函数

```python
import pandas as pd
import numpy as np

data = {'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
        'Value': [10, np.nan, 20, 25, 12, np.nan]}
df = pd.DataFrame(data)
display(df)
# 用每个 Category 组的均值填充 Value 列的缺失值
df['Value_Filled'] = df.groupby('Category')['Value'].transform(lambda x: x.fillna(x.mean()))
display(df)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250228132631439.png" alt="image-20250228132631439" style="zoom:50%;" />

# **计算组内的排名 (Ranking)**

> 练习 力扣：[184. 部门工资最高的员工](https://leetcode.cn/problems/department-highest-salary/)

```python
import pandas as pd

data = {'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
        'Value': [10, 15, 20, 25, 12, 22]}
df = pd.DataFrame(data)
display(df)
# 计算每个 Category 组内 Value 列的排名
df['Value_Rank'] = df.groupby('Category')['Value'].transform(lambda x: x.rank())
display(df)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250228132754738.png" alt="image-20250228132754738" style="zoom:50%;" />

# **创建滞后 (Lagged) 或超前 (Lead) 值**

使用 shift() 函数结合 transform()

```python
import pandas as pd

data = {'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
        'Value': [10, 15, 20, 25, 12, 22]}
df = pd.DataFrame(data)
display(df)
# 创建每个 Category 组内 Value 列的滞后 1 期值
df['Value_Lagged'] = df.groupby('Category')['Value'].transform(lambda x: x.shift(1))
display(df)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250228133048305.png" alt="image-20250228133048305" style="zoom:50%;" />
