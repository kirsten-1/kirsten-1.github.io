---
layout: post
title: "numpy(6)深入学习NumPy模块"
subtitle: "第 6 章 深入学习NumPy模块"
date: 2025-02-09
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 人工智能AI基础
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>


前5章以及其他补充已经整理如下：

[第1章NumPy入门](https://kirsten-1.github.io/2025/01/27/numpy(1)_%E5%85%A5%E9%97%A8/)

[第2章NumPy基础](https://kirsten-1.github.io/2025/02/04/numpy(2)_numpy%E5%9F%BA%E7%A1%80/)

[第3章常用函数](https://kirsten-1.github.io/2025/02/06/Numpy(3)%E5%B8%B8%E7%94%A8%E5%87%BD%E6%95%B0/)

[广播与广播机制](https://kirsten-1.github.io/2025/02/04/Numpy%E7%9A%84%E5%B9%BF%E6%92%AD%E5%92%8C%E5%B9%BF%E6%92%AD%E6%9C%BA%E5%88%B6/)

[第4章便捷函数](https://kirsten-1.github.io/2025/02/07/NumPy(4)%E4%BE%BF%E6%8D%B7%E5%87%BD%E6%95%B0/)

[第5章矩阵和通用函数](https://kirsten-1.github.io/2025/02/09/NumPy(5)%E7%9F%A9%E9%98%B5%E5%92%8C%E9%80%9A%E7%94%A8%E5%87%BD%E6%95%B0/)

----

NumPy中有很多模块是从它的前身Numeric继承下来的。这些模块有一部分在SciPy中也有对应的部分，并且功能可能更加丰富，我们将在后续章节中讨论相关内容。`numpy.dual`模块包含同时在NumPy和SciPy中定义的函数。在本章中讨论的模块也属于`numpy.dual`的一部分。

> `SciPy`（科学计算库）是 Python 中一个用于数学、科学和工程计算的库。它是基于 NumPy 构建的，并提供了许多高级数学函数和算法。它的名称 `SciPy` 是 `Scientific Python`（科学 Python）的缩写。

本章涵盖以下内容：

- linalg模块；
- fft模块；
- 随机数；
- 连续分布和离散分布。

> 注：`linalg` 模块：
>
> - **全称**：Linear Algebra（线性代数）
> - **功能**：`linalg` 模块提供了许多常用的线性代数运算功能，包括矩阵乘法、求解线性方程组、特征值分解、奇异值分解等。
>
> `fft` 模块：
>
> - **全称**：Fast Fourier Transform（快速傅里叶变换）
> - **功能**：`fft` 模块提供了傅里叶变换相关的功能，特别是快速傅里叶变换（FFT），用于计算信号在频域中的表示。

---

# 6.1 线性代数

线性代数是数学的一个重要分支。`numpy.linalg`模块包含线性代数的函数。使用这个模块，我们可以计算**逆矩阵、求特征值、解线性方程组以及求解行列式**等。

# 6.2 动手实践：计算逆矩阵

在线性代数中，矩阵 $$ A $$ 与其逆矩阵$$ A^{-1} $$相乘后会得到一个单位矩阵$$I$$。该定义可以写为$$A * A^{-1}  =I$$。`numpy.linalg`模块中的**inv函数**可以计算逆矩阵。我们按如下步骤来对矩阵求逆。

(1) 与前面的教程中一样，我们将使用mat函数创建示例矩阵：

```python
m = np.mat('0 1 2; 1 0 3; 4 -3 8')
m
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209160346122.png" alt="image-20250209160346122" style="zoom:50%;" />

(2) 现在，我们使用inv函数计算逆矩阵：

```python
inverse = np.linalg.inv(m)
inverse
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209160446101.png" alt="image-20250209160446101" style="zoom:50%;" />

如果输入矩阵是奇异的或非方阵，则会抛出`LinAlgError`异常。我们将此作为练习留给读者，如果你愿意，可以动手尝试一下。

例如：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209160637146.png" alt="image-20250209160637146" style="zoom:50%;" />

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209160721442.png" alt="image-20250209160721442" style="zoom:50%;" />

(3) 我们来检查一下原矩阵和求得的逆矩阵相乘的结果：

```python
m*inverse
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209160754284.png" alt="image-20250209160754284" style="zoom:50%;" />

刚才做了些什么 : 我们使用`numpy.linalg`模块中的**inv函数**计算了逆矩阵，并检查了原矩阵与求得的逆矩阵相乘的结果确为单位矩阵。

完整代码如下：

```python
import numpy as np

m = np.mat('0 1 2; 1 0 3; 4 -3 8')
inverse = np.linalg.inv(m)
print(m)
print(inverse)
print(m * inverse)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209160912159.png" alt="image-20250209160912159" style="zoom:50%;" />

## 突击测验：如何创建矩阵

问题1 以下哪个函数可以创建矩阵？

(1) array (2) create_matrix (3)  mat (4) vector

> 答案：（3）在 SciPy 和 NumPy 中，创建矩阵的函数是：(3) mat
>
> **`array`**：这个函数可以创建 NumPy 数组，不仅仅是矩阵。它是 NumPy 中最常用的创建数组的方法。
>
> **`create_matrix`**：这不是一个有效的函数名，NumPy 或 SciPy 中并没有这样一个函数。
>
> **`vector`**：`vector` 不是一个有效的函数名，因此不能用来创建矩阵。

## 勇敢出发：创建新矩阵并计算其逆矩阵

请自行创建一个新的矩阵并计算其逆矩阵。注意，你的矩阵必须是方阵且可逆，否则会抛出LinAlgError异常。

> 略

# 6.3 求解线性方程组

矩阵可以对向量进行线性变换，这对应于数学中的线性方程组。`numpy.linalg`中的函数solve可以求解形如$$ Ax = b $$的线性方程组，其中 $$A$$ 为矩阵，$$b$$ 为一维或二维的数组，$$x$$ 是未知变量。我们将练习使用**dot函数**，用于计算两个浮点数数组的点积。

# 6.4 动手实践：求解线性方程组

让我们求解一个线性方程组实例，步骤如下。

(1) 创建矩阵$$A$$和数组$$b$$：

```python
A = np.mat('1 -2 1;0 2 -8;-4 5 9')
print("A", A)
b = np.array([0, 8, -9])
print("b", b)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209161335645.png" alt="image-20250209161335645" style="zoom:50%;" />

(2) 调用solve函数求解线性方程组：

```python
solution = np.linalg.solve(A, b)
print("solution", solution)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209161439515.png" alt="image-20250209161439515" style="zoom:50%;" />

(3) 使用dot函数检查求得的解是否正确：

```python
print("A * x", np.dot(A, solution))
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209161536978.png" alt="image-20250209161536978" style="zoom:50%;" />

刚才做了些什么 :  我们使用NumPy的linalg模块中的solve函数求解了线性方程组，并使用dot函数验证了求解的正确性。

完整代码如下：

```python
A = np.mat('1 -2 1;0 2 -8;-4 5 9')
print("A", A)
b = np.array([0, 8, -9])
print("b", b)
solution = np.linalg.solve(A, b)
print("solution", solution)
print("A * x", np.dot(A, solution))
```

# 6.5 特征值和特征向量

特征值（eigenvalue）即方程$$ Ax = ax$$ 的根，是一个标量。其中，$$A$$ 是一个二维矩阵，$$x$$ 是一个一维向量。特征向量（eigenvector）是关于特征值的向量。**在`numpy.linalg`模块中，`eigvals `函数可以计算矩阵的特征值，而`eig`函数可以返回一个包含特征值和对应的特征向量的元组。**

补充：

在 **NumPy** 中，`eigvals()` 和 `eig()` 都是用于计算矩阵的特征值和特征向量的函数。它们分别具有不同的功能和返回结果。

---

`numpy.linalg.eigvals()` 函数：`eigvals()` 用于计算一个方阵的 **特征值**。特征值是与矩阵相关的标量，表示矩阵的某些性质。`eigvals()` 只返回特征值，不返回特征向量。

函数签名：

```python
numpy.linalg.eigvals(a)
```

**a**：一个方阵（2D 数组或矩阵），用于计算其特征值。

返回值:返回一个包含特征值的 1D 数组。

```python
A = np.mat("4 -2; 1 1")
np.linalg.eigvals(A)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209162036700.png" alt="image-20250209162036700" style="zoom:50%;" />

---

`numpy.linalg.eig()` 函数:`eig()` 函数用于计算一个方阵的 **特征值** 和 **特征向量**。它返回一个包含特征值和特征向量的元组。

函数签名：

```python
numpy.linalg.eig(a)
```

- **a**：一个方阵（2D 数组或矩阵），用于计算其特征值和特征向量。

返回值：返回一个元组 `(w, v)`：

- **w**：包含特征值的 1D 数组。
- **v**：包含特征向量的二维数组，每列是对应特征值的特征向量。

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues, eigenvectors
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209162226952.png" alt="image-20250209162226952" style="zoom:50%;" />

----

# 6.6 动手实践：求解特征值和特征向量

我们来计算矩阵的特征值和特征向量，步骤如下。

(1) 创建一个矩阵：

```python
A = np.mat("3 -2;1 0")
print(A)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209162332131.png" alt="image-20250209162332131" style="zoom:50%;" />

(2) 调用eigvals函数求解特征值：

```python
np.linalg.eigvals(A)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209162408403.png" alt="image-20250209162408403" style="zoom:50%;" />

(3) 使用eig函数求解特征值和特征向量。该函数将返回一个元组，按列排放着特征值和对应的特征向量，其中第一列为特征值，第二列为特征向量。

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues, eigenvectors
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209162513506.png" alt="image-20250209162513506" style="zoom:50%;" />

(4) 使用dot函数验证求得的解是否正确。分别计算等式 $$Ax = ax$$ 的左半部分和右半部分，检查是否相等。

```python
# Ax = ax
for i in range(len(eigenvalues)):
    print(np.dot(A, eigenvectors[:, -i]))
    print(eigenvalues[i] * eigenvectors[:,i] )
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209163013040.png" alt="image-20250209163013040" style="zoom:50%;" />

注意：验证方法不是这样的：

```python
# Ax = ax
print(A * eigenvectors)
print(eigenvalues * eigenvectors)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209162652589.png" alt="image-20250209162652589" style="zoom:50%;" />

为什么？注意`eigenvectors[:, -i]`代表什么：

```python
print(eigenvectors)
print("-------")
for i in range(len(eigenvalues)):
    print(eigenvectors[:, -i])# 代表列向量
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209163150429.png" alt="image-20250209163150429" style="zoom:50%;" />

例如，`eigenvectors[:, -1]` 选择的是 `eigenvectors` 的最后一列，即对应矩阵最后一个特征值的特征向量；而 `eigenvectors[:, -2]` 选择的是倒数第二列，即对应第二个特征值的特征向量。

这种索引方式常用于需要访问矩阵中的特定列（例如特征向量）时，尤其是在不知道具体列索引或列数的情况下，负索引可以帮助我们灵活地从矩阵的末尾访问元素。

---

刚才做了些什么 : 我们使用`numpy.linalg`模块中的`eigvals`和`eig`函数求解了矩阵的特征值和特征向量，并使用`dot`函数进行了验证。

完整代码如下：

```python
import numpy as np

A = np.mat("3 -2;1 0")
print(np.linalg.eigvals(A))
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues, eigenvectors)
#验证
# Ax = ax
for i in range(len(eigenvalues)):
    print(np.dot(A, eigenvectors[:, -i]))
    print(eigenvalues[i] * eigenvectors[:,i] )
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209164806008.png" alt="image-20250209164806008" style="zoom:50%;" />

# 6.7 奇异值分解

SVD（Singular Value Decomposition，奇异值分解）是一种因子分解运算，将一个矩阵分解为3个矩阵的乘积。奇异值分解是前面讨论过的特征值分解的一种推广。在`numpy.linalg`模块中的**svd函数**可以对矩阵进行奇异值分解。该函数返回3个矩阵——`U`、`Sigma`和`V`，其中`U`和`V`是正交矩阵，`Sigma`包含输入矩阵的奇异值。


$$
M = U \sum V^{*}
$$


星号表示厄米共轭（Hermitian conjugate）或共轭转置（conjugate transpose）。

----

补充：svd函数的使用方法

SVD 是线性代数中的一种常见分解方法，用于将任意矩阵分解为三个矩阵的乘积。它在许多应用中都非常重要，比如数据压缩、特征提取和求解线性方程等。

函数签名：

```python
numpy.linalg.svd(a, full_matrices=True, compute_uv=True)
```

参数:

- **a**：输入矩阵（二维数组或矩阵），通常是一个 $$m \times n$$的矩阵。
- full_matrices：布尔值，控制是否计算完整的奇异值分解。
    - 如果为 `True`（默认），则返回完整的 $$U$$ 和 $$V$$ 矩阵，尺寸分别是 $$m \times m$$ 和$$n \times n$$。
    - 如果为 `False`，则返回紧凑形式，$$U$$ 和 $$V^T$$ 将是 $$m \times \min(m, n)$$ 和 $$\min(m, n) \times n$$。
- compute_uv：布尔值，控制是否计算 $$U$$ 和 $$V$$矩阵。
    - 如果为 `True`（默认），则返回 $$U$$, $$\Sigma$$和 $$V^T$$。
    - 如果为 `False`，则只计算奇异值 $$\Sigma$$。

返回值:

- **U**：包含**左奇异向量**的矩阵。
- **S**：包含奇异值的一维数组。注意，`S` 是一个包含非负奇异值的数组，但它不是对角矩阵。如果需要对角矩阵，可以使用 `np.diag(S)` 转换为对角矩阵。
- **Vh**（即 $$V^T$$）：包含**右奇异向量**的转置矩阵。

例子：

```python
import numpy as np

# 创建一个示例矩阵 A
A = np.array([[1, 2], [3, 4], [5, 6]])

# 使用 svd 函数计算 SVD
U, S, Vh = np.linalg.svd(A)

# 输出结果
print("U:")
print(U)
print("\n奇异值 S:")
print(S)
print("\nVh (V的转置):")
print(Vh)

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209165541818.png" alt="image-20250209165541818" style="zoom:50%;" />

# 6.8 动手实践：分解矩阵

现在，我们来对矩阵进行奇异值分解，步骤如下。

(1) 首先，创建一个矩阵：

```python
A = np.mat('4 11 14;8 7 -2')
A
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209165632792.png" alt="image-20250209165632792" style="zoom:50%;" />

(2) 使用`svd`函数分解矩阵：

```python
U, Sigma, V = np.linalg.svd(A, full_matrices=False)
U, Sigma, V
```

得到的结果包含等式中左右两端的两个正交矩阵U和V，以及中间的奇异值矩阵Sigma：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209165808339.png" alt="image-20250209165808339" style="zoom:50%;" />

(3) 不过，我们并没有真正得到中间的奇异值矩阵——得到的只是其对角线上的值，而非对角线上的值均为0。我们可以使用diag函数生成完整的奇异值矩阵。将分解出的3个矩阵相乘，如下所示：

```python
U * np.diag(Sigma) * V
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209165927162.png" alt="image-20250209165927162" style="zoom:50%;" />

刚才做了些什么 : 我们分解了一个矩阵，并使用矩阵乘法验证了分解的结果。我们使用了NumPy linalg模块中的svd函数。

完整代码如下：

```python
import numpy as np
A = np.mat('4 11 14;8 7 -2')
U, Sigma, V = np.linalg.svd(A, full_matrices=False)
print("\nA:", A)
print("\nU:", U)
print("\nSigma:",Sigma)
print("\nV:", V)
# 验证
print("\n验证结果U * Sigma * V:",U * np.diag(Sigma) * V)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209170235450.png" alt="image-20250209170235450" style="zoom:50%;" />

# 6.9 广义逆矩阵

摩尔·彭罗斯广义逆矩阵（Moore-Penrose pseudoinverse）可以使用`numpy.linalg`模块中的pinv函数进行求解（广义逆矩阵的具体定义请访问http://en.wikipedia.org/wiki/Moore%E2%80%-93Penrose_pseudoinverse）。**计算广义逆矩阵需要用到奇异值分解。inv函数只接受方阵作为输入矩阵，而pinv函数则没有这个限制**。

----

补充：pinv函数的使用方法

广义逆是通过对矩阵进行某些特定的操作来定义的，它对非方阵和奇异矩阵非常有用。例如，在求解线性方程组时，当系统没有唯一解或者没有解时，广义逆提供了一种最小化误差的“最佳”解。

函数签名：

```python
numpy.linalg.pinv(a, rcond=1e-15)
```

参数:

- **a**：输入矩阵（2D 数组或矩阵），可以是**方阵或非方阵**。
- **rcond**：截断奇异值的阈值，**表示奇异值的最小值**。当奇异值小于该值时，它们将被视为零。`rcond` 的默认值是 `1e-15`，即在计算广义逆时，低于该值的奇异值将被视为零。

返回值：返回矩阵 **a** 的摩尔-彭若斯广义逆（伪逆），它具有与原矩阵形状相关的维度。

```python
import numpy as np

# 创建一个矩阵 A
A = np.array([[1, 2], [3, 4]])

# 计算 A 的摩尔-彭若斯广义逆
A_pinv = np.linalg.pinv(A)

print("原矩阵 A:")
print(A)
print("\n广义逆 A_pinv:")
print(A_pinv)
# 验证
print("\n验证：", A * A_pinv)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209171556911.png" alt="image-20250209171556911" style="zoom:50%;" />

应用：假设我们有一个线性方程组 `Ax = b`，如果 `A` 是一个不可逆矩阵（例如行列式为零），我们可以使用广义逆来找到一个最小二乘解。

```python
# 创建一个不可逆矩阵 A 和向量 b
A = np.array([[1, 2], [2, 4]])  # 行列式为零
b = np.array([5, 10])

# 使用广义逆来求解 x
x = np.linalg.pinv(A).dot(b)

print("解 x:")
print(x)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209171624877.png" alt="image-20250209171624877" style="zoom:50%;" />

# 6.10 动手实践：计算广义逆矩阵

我们来计算矩阵的广义逆矩阵，步骤如下。

(1) 首先，创建一个矩阵：

```python
A = np.mat('4 11 14;8 7 -2')
A
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209171715780.png" alt="image-20250209171715780" style="zoom:50%;" />

(2) 使用pinv函数计算广义逆矩阵：

```python
pseudoinv = np.linalg.pinv(A)
pseudoinv
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209171755824.png" alt="image-20250209171755824" style="zoom:50%;" />

(3) 将原矩阵和得到的广义逆矩阵相乘：

```python
A * pseudoinv
```

得到的结果并非严格意义上的单位矩阵，但非常近似，如下所示：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209171916336.png" alt="image-20250209171916336" style="zoom:50%;" />

刚才做了些什么 :   我们使用`numpy.linalg`模块中的pinv函数计算了矩阵的广义逆矩阵。在验证时，用原矩阵与广义逆矩阵相乘，得到的结果为一个近似单位矩阵。

完整代码如下：

```python
import numpy as np

A = np.mat('4 11 14;8 7 -2')
pseudoinv = np.linalg.pinv(A)

print("\nA:", A)
print("\npseudoinv: ", pseudoinv)
print("\ncheck:", A * pseudoinv)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209172141881.png" alt="image-20250209172141881" style="zoom:50%;" />

# 6.11 行列式

行列式（determinant）是与方阵相关的一个标量值，在数学中得到广泛应用（更详细的介绍请访问http://en.wikipedia.org/wiki/Determinant）。

对于一个$$n×n$$的实数矩阵，行列式描述的是一个线性变换对“有向体积”所造成的影响。行列式的值为正表示保持了空间的定向（顺时针或逆时针），为负则表示颠倒了空间的定向。`numpy.linalg`模块中的**det函数**可以计算矩阵的行列式。

---

补充：det函数的使用方法

在 **NumPy** 的 `linalg` 模块中，`det()` 函数用于计算一个方阵的 **行列式（determinant）**。行列式是一个标量值，用于描述矩阵的某些特性。特别地，行列式可以用来判断一个矩阵是否是可逆的（非零行列式的矩阵是可逆的）。

函数签名：

```python
numpy.linalg.det(a)
```

**参数（输入是方针！！）**

- **a**：输入矩阵（必须是二维的方阵），通常是一个 $$n \times n$$的方阵。

返回值：

- 返回矩阵 `a` 的行列式，类型为标量（float）。

例如：

```python
import numpy as np

# 创建一个 2x2 方阵
A = np.array([[4, 2],
              [3, 1]])

# 计算行列式
det_A = np.linalg.det(A)

print("矩阵 A 的行列式:", det_A)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209172501201.png" alt="image-20250209172501201" style="zoom:50%;" />

----

# 6.12 动手实践：计算矩阵的行列式

计算矩阵的行列式，步骤如下。

(1) 创建一个矩阵：

```python
A = np.mat('3 4;5 6')
A
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209172545400.png" alt="image-20250209172545400" style="zoom:50%;" />

(2) 使用det函数计算行列式：

```python
np.linalg.det(A)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209172624013.png" alt="image-20250209172624013" style="zoom:50%;" />

刚才做了些什么 :  我们使用`numpy.linalg`模块中的det函数计算了矩阵的行列式。

---

# 6.13 快速傅里叶变换

FFT（Fast Fourier Transform，快速傅里叶变换）是一种高效的计算DFT（Discrete Fourier Transform，离散傅里叶变换）的算法。

FFT算法比根据定义直接计算更快，计算复杂度为 O(NlogN) 。

DFT在信号处理、图像处理、求解偏微分方程等方面都有应用。

在NumPy中，有一个名为fft的模块提供了快速傅里叶变换的功能。在这个模块中，许多函数都是成对存在的，也就是说许多函数存在对应的逆操作函数。例如，**fft和ifft函数**就是其中的一对。

> 关于傅立叶变换，推荐看视频：https://www.bilibili.com/video/BV1aW4y1y7Hs/?spm_id_from=333.337.search-card.all.click&vd_source=6c6e2754e61f483e81b4bc03c9898c87
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209173453913.png" alt="image-20250209173453913" style="zoom:50%;" />
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209174346170.png" alt="image-20250209174346170" style="zoom:50%;" />
>
> 打算看《信号与系统》这本书，补充以下这块的知识点。



---

# 6.14 动手实践：计算傅里叶变换

首先，我们将创建一个信号用于变换。计算傅里叶变换的步骤如下。

(1) 创建一个包含30个点的余弦波信号，如下所示：

```python
x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
wave
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209191309229.png" alt="image-20250209191309229" style="zoom:50%;" />

(2) 使用fft函数对余弦波信号进行傅里叶变换。

```python
transformed = np.fft.fft(wave)
transformed
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209191407326.png" alt="image-20250209191407326" style="zoom:50%;" />

(3) 对变换后的结果应用ifft函数，应该可以近似地还原初始信号。

```python
np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209191634175.png" alt="image-20250209191634175" style="zoom:50%;" />

(4) 使用Matplotlib绘制变换后的信号。

```python
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

plot(transformed)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209191748994.png" alt="image-20250209191748994" style="zoom:50%;" />

刚才做了些什么 : 我们在余弦波信号上应用了**fft函数**，随后又对变换结果应用**ifft函数**还原了信号。

完整代码：

```python
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)

# 傅立叶变换
transformed = np.fft.fft(wave)

# 逆变换
print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))

# 绘图
plot(transformed)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209192221674.png" alt="image-20250209192221674" style="zoom:50%;" />

# 6.15 移频

`numpy.linalg`模块中的**fftshift函数**可以将FFT输出中的直流分量移动到频谱的中央。**ifftshift函数**则是其逆操作。

# 6.16 动手实践：移频

我们将创建一个信号用于变换，然后进行移频操作，步骤如下。

(1) 创建一个包含30个点的余弦波信号。

```python
x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
```

(2) 使用fft函数对余弦波信号进行傅里叶变换。

```python
transformed = np.fft.fft(transformed)
```

(3) 使用fftshift函数进行移频操作。

```python
shifted = np.fft.fftshift(transformed)
```

(4) 用ifftshift函数进行逆操作，这将还原移频操作前的信号。

```python
np.all((np.fft.ifftshift(shifted) - transformed) < 10 ** -9)
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209192651791.png" alt="image-20250209192651791" style="zoom:50%;" />

(5) 使用Matplotlib分别绘制变换和移频处理后的信号。

```python
plot(transformed, lw = 2)
plot(shifted, lw = 3)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209192817018.png" alt="image-20250209192817018" style="zoom:50%;" />

刚才做了些什么 : 我们在傅里叶变换后的余弦波信号上应用了fftshift函数，随后又应用ifftshift函数还原了信号。

完整代码：

```python
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)

transformed = np.fft.fft(wave)

shifted = np.fft.fftshift(transformed)
print(np.all((np.fft.ifftshift(shifted) - transformed) < 10 ** -9))

plot(transformed, lw = 2)
plot(shifted, lw = 3)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209200300821.png" alt="image-20250209200300821" style="zoom:50%;" />

# 6.17 随机数

随机数在蒙特卡罗方法（Monto Carlo method）、随机积分等很多方面都有应用。真随机数的产生很困难，因此在实际应用中我们通常使用**伪随机数**。在大部分应用场景下，伪随机数已经足够随机，当然一些特殊应用除外。

有关随机数的函数可以在NumPy的random模块中找到。随机数发生器的核心算法是基于**马特赛特旋转演算法（Mersenne Twister algorithm）**的。随机数可以从**离散分布或连续分布**中产生。分布函数有一个可选的参数**size**，用于指定需要产生的随机数的数量。该参数允许设置为一个整数或元组，生成的随机数将填满指定形状的数组。支持的离散分布包括**几何分布、超几何分布和二项分布**等。

---

# 6.18 动手实践：硬币赌博游戏

二项分布是n个独立重复的是/非试验中成功次数的离散概率分布，这些概率是固定不变的，与试验结果无关。

设想你来到了一个17世纪的赌场，正在对一个硬币赌博游戏下8份赌注。每一轮抛9枚硬币，如果少于5枚硬币正面朝上，你将损失8份赌注中的1份；否则，你将赢得1份赌注。我们来模拟一下赌博的过程，初始资本为1000份赌注。为此，我们需要使用random模块中的`binomial `函数。

---

补充：`np.random.binomial`的使用方法

`np.random.binomial()` 是 NumPy 中用于生成 **二项分布** 随机变量的函数。二项分布用于描述在相同条件下进行多次独立的伯努利试验（即只有两种结果的试验，如成功/失败）时，成功次数的概率分布。

$$n$$次独立的试验中，成功的次数$$k$$的概率，概率质量函数为：$$P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$$

函数签名：

```python
numpy.random.binomial(n, p, size=None)
```

参数

- **n**：每次实验的试验次数（正整数）。
- **p**：每次实验成功的概率（介于 0 和 1 之间的浮动数）。
- **size**：生成的随机数的形状。如果是一个整数，返回该数量的样本；如果是一个元组，返回形状为元组的随机数组。

返回值：返回一个整数或一个整数数组，表示在每次试验中成功的次数。

----

为了理解binomial函数的用法，请完成如下步骤。

(1) 初始化一个全0的数组来存放剩余资本。以参数10000调用binomial函数，意味着我们将在赌场中玩10 000轮硬币赌博游戏。

```python
# 创建一个长度为 10000 的数组，用于存储每轮后的剩余资本。所有元素初始值为 0。
cash = np.zeros(10000)
# 将初始资本设置为 1000 份赌注。
cash[0] = 1000
# 模拟每轮抛 9 枚硬币，成功概率为 0.5。每个结果是一个整数，表示该轮实验中正面朝上的硬币数量（范围 0 到 9）。
outcome = np.random.binomial(9, 0.5, size=len(cash))
outcome
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209203021091.png" alt="image-20250209203021091" style="zoom:50%;" />

(2) 模拟每一轮抛硬币的结果并更新cash数组。打印出outcome的最小值和最大值，以检查输出中是否有任何异常值：

```python
for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i-1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i-1] + 1
    else:
        raise AssertionError("Unexpected outcome " + outcome) 
print("最小：", cash.min())
print("最大：", cash.max())
print("最小次数：", outcome.min())
print("最大次数：", outcome.max())
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209203341906.png" alt="image-20250209203341906" style="zoom:50%;" />

(3) 使用Matplotlib绘制cash数组：

```python
plot(np.arange(len(cash)), cash)
show()
```

注，反复执行`np.random.binomial`函数，绘制的图形不一样。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209203542662.png" alt="image-20250209203542662" style="zoom:50%;" />

刚才做了些什么 ：我们使用NumPy random模块中的binomial函数模拟了随机游走。

完整代码如下：

```python
import numpy 
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

cash = np.zeros(10000)
cash[0] = 1000
print("初始资金：", cash)

outcome = np.random.binomial(9, 0.5, size = len(cash))
for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i-1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i-1] + 1
    else:
        raise AssertionError("outcome异常：", outcome[i])

print("最大的次数：", np.max(outcome))
print("最小的次数：", np.min(outcome))
print("钱最多的时候", np.max(cash))
print("钱最少的时候", np.min(cash))

#画图
plot(np.arange(len(cash)), cash)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209204156206.png" alt="image-20250209204156206" style="zoom:50%;" />

# 6.19 超几何分布

超几何分布（hypergeometric distribution）是一种离散概率分布，它描述的是一个罐子里有两种物件，**无放回地**从中抽取指定数量的物件后，抽出指定种类物件的数量。NumPy random模块中的hypergeometric函数可以模拟这种分布。

---

补充：超几何分布

例如：从一副 52 张扑克牌中抽 5 张牌，计算抽到 3 张红心的概率，或者从一批不合格和合格产品中抽样检验不合格品的数量。

超几何分布的概率质量函数为：$$P(X = k) = \frac{\binom{K}{k}\binom{N - k}{n - k}}{\binom{N}{n}}$$

- $$N$$是总体的大小（总的元素数量）。
- $$K$$是总体中成功元素的数量（例如，总体中合格产品的数量）。
- $$n$$是抽样的大小（即每次抽取的元素数目）。
- $$k$$是抽样中成功元素的数量（即我们关心的特定类别的数量）。
- $$\binom{a}{b}$$表示从 $$a$$ 个元素中选择 $$b$$ 个元素的组合数。

----

补充：NumPy random模块中的hypergeometric函数的使用方法

在 **NumPy** 中，`random` 模块提供了一个 **`hypergeometric()`** 函数，用于生成符合 **超几何分布** 的随机数。超几何分布描述的是在**没有放回的情况下**，从有限总体中抽取一定数量的成功元素的概率分布。

函数签名：

```python
numpy.random.hypergeometric(low, high, size, shape=None)
```

参数：

- **low**：成功元素的数量 $$K$$。
- **high**：失败元素的数量 $$N-K$$。
- **size**：每次实验中抽取的样本大小 $$n$$。
- **shape**：返回的输出数组的形状，默认返回一个标量或数组。

返回值：

- 返回的是一个数组，数组中的每个值表示每次实验中成功的元素数量。

假设我们有一个装有 52 张牌的扑克牌，其中 13 张是红心，39 张是非红心。我们从中随机抽取 5 张牌，计算其中有多少张红心。

```python
import numpy as np

# 设定参数
low = 13  # 红心牌的数量 (成功元素数量 K)
high = 39  # 非红心牌的数量 (失败元素数量 N - K)
n = 5  # 抽取 5 张牌

# 从 52 张牌中随机抽取 5 张，计算其中红心牌的数量
result = np.random.hypergeometric(low, high, n)

print("抽到的红心牌数量：", result)

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209205030759.png" alt="image-20250209205030759" style="zoom:50%;" />

----

# 6.20 动手实践：模拟游戏秀节目

设想有这样一个游戏秀节目，每当参赛者回答对一个问题，他们可以从一个罐子里摸出3个球并放回。罐子里有一个“倒霉球”，一旦这个球被摸出，参赛者会被扣去6分。而如果他们摸出的3个球全部来自其余的25个普通球，那么可以得到1分。因此，如果一共有100道问题被正确回答，得分情况会是怎样的呢？

为了解决这个问题，请完成如下步骤。

(1) 使用hypergeometric函数初始化游戏的结果矩阵。该函数的第一个参数为罐中普通球的数量，第二个参数为“倒霉球”的数量，第三个参数为每次采样（摸球）的数量。

```python
points = np.zeros(100)
outcomes = np.random.hypergeometric(25, 1, 3, len(points))
outcomes
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209205206798.png" alt="image-20250209205206798" style="zoom:50%;" />

(2) 根据上一步产生的游戏结果计算相应的得分。

```python
for i in range(len(points)):
    if outcomes[i] == 3:
        points[i] = points[i-1] + 1
    elif outcomes[i] == 2:
        points[i] = points[i-1] - 6
    else:
        print("异常情况：", points[i])
        
points
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209205443478.png" alt="image-20250209205443478" style="zoom:50%;" />

(3) 使用Matplotlib绘制points数组。

```python
plot(np.arange(len(points)), points)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209205535650.png" alt="image-20250209205535650" style="zoom:50%;" />

刚才做了些什么 : 我们使用NumPy random模块中的hypergeometric函数模拟了一个游戏秀节目。这个游戏的得分取决于每一轮从罐子里摸出的球的种类。

完整代码如下：

```python
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

points = np.zeros(100)
outcomes = np.random.hypergeometric(25, 1, 3, size=len(points))
print("每次抽取结果：", outcomes)

# 计算得分
for i in range(len(points)):
    if outcomes[i] == 3:
        points[i] = points[i-1] + 1
    elif outcomes[i] == 2:
        points[i] = points[i-1] - 6
    else:
        print("异常情况：", outcomes[i])

print("每轮得分情况：", points)

# 画图
plot(np.arange(len(points)), points)
show()
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209210117423.png" alt="image-20250209210117423" style="zoom:50%;" />

# 6.21 连续分布

连续分布可以用PDF（Probability Density Function，概率密度函数）来描述。随机变量落在某一区间内的概率等于概率密度函数在该区间的曲线下方的面积。NumPy的random模块中有一系列连续分布的函数——`beta、chisquare、exponential、f、gamma、gumbel、laplace、lognormal、 logistic、multivariate_normal、noncentral_chisquare、noncentral_f、normal`等。

# 6.22 动手实践：绘制正态分布

随机数可以从正态分布中产生，它们的直方图能够直观地刻画正态分布。按照如下步骤绘制正态分布。

(1) 使用NumPy random模块中的normal函数产生指定数量的随机数。

```python
N = 100
normal_values = np.random.normal(size = N)
normal_values
```

(2) 绘制分布直方图和理论上的概率密度函数（均值为0、方差为1的正态分布）曲线。我们将使用Matplotlib进行绘图。

```python
import numpy as np
import matplotlib.pyplot as plt

# (1) 使用 NumPy 生成指定数量的正态分布随机数
N = 10000
normal_values = np.random.normal(size=N)

# (2) 绘制分布直方图和理论上的概率密度函数（均值为0、方差为1的正态分布）曲线
bins = int(np.sqrt(N))  # 将箱子数设置为 sqrt(N) 并转为整数

# 绘制直方图， density=True 表示将直方图标准化为概率密度
dummy, bins, dummy = plt.hist(normal_values, bins=bins, density=True, lw=1, alpha=0.6, color='g')

# 正态分布的参数
sigma = 1  # 标准差
mu = 0  # 均值

# 绘制理论上的正态分布曲线
x = np.linspace(min(bins), max(bins), 1000)  # 创建一个用于绘制曲线的 x 值范围
pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))  # 正态分布的概率密度函数公式
plt.plot(x, pdf, lw=2, label="Theoretical PDF", color='r')

# 显示图例
plt.legend()

# 显示图形
plt.show()

```

注：原书代码，在现在的matplotlib中已经不适用了。在新版的 `matplotlib` 中，`normed` 已被弃用，取而代之的是 `density=True`。`density=True` 会将直方图的高度归一化，使其显示概率密度而非频数。

所以上面的代码是完整修改过后的代码.

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209210813061.png" alt="image-20250209210813061" style="zoom:50%;" />

刚才做了些什么 : 我们画出了NumPy random模块中的normal函数模拟的正态分布。我们将该函数生成的随机数绘制成分布直方图，并同时绘制了标准正态分布的钟形曲线。

# 6.23 对数正态分布

对数正态分布（lognormal distribution） 是自然对数服从正态分布的任意随机变量的概率分布。NumPy random模块中的lognormal函数模拟了这个分布。

# 6.24 动手实践：绘制对数正态分布

我们绘制出对数正态分布的概率密度函数以及对应的分布直方图，步骤如下。

(1) 使用NumPy random模块中的normal函数产生随机数。

```python
N = 10000
lognormal_values = np.random.lognormal(size = N)
```

(2) 绘制分布直方图和理论上的概率密度函数（均值为0、方差为1）。我们将使用Matplotlib进行绘图。

注：原书代码会报错。问题和上面一样，在新版 `matplotlib` 中，`normed` 参数已经被弃用，改为 `density=True`，用来标准化直方图，使其显示为概率密度而不是频数。

下面是修改后的完整的代码：

```python
import numpy as np
import matplotlib.pyplot as plt

# (1) 使用 NumPy 生成对数正态分布的随机数
N = 10000
lognormal_values = np.random.lognormal(size=N)

# (2) 绘制对数正态分布的直方图和理论概率密度函数（PDF）
bins = int(np.sqrt(N))  # 设置箱子数量为 sqrt(N)，并转为整数

# 绘制对数正态分布的直方图，density=True 标准化为概率密度
dummy, bins, dummy = plt.hist(lognormal_values, bins=bins, density=True, lw=1, alpha=0.6, color='g')

# 对数正态分布的参数
sigma = 1  # 标准差
mu = 0  # 均值

# 创建用于绘制 PDF 的 x 轴范围
x = np.linspace(min(bins), max(bins), 1000)

# 对数正态分布的 PDF 公式
pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

# 绘制理论的对数正态分布的 PDF 曲线
plt.plot(x, pdf, lw=3, color='r', label="Theoretical PDF")

# 显示图例
plt.legend()

# 显示图形
plt.show()

```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250209211129666.png" style="zoom:50%;" />

直方图和理论概率密度函数的曲线吻合得很好。

刚才做了些什么 ：我们画出了NumPy random模块中的lognormal函数模拟的对数正态分布。我们将该函数生成的随机数绘制成分布直方图，并同时绘制了理论上的概率密度函数曲线。

# 6.25 本章小结

在本章中，我们学习了很多NumPy模块的知识，涵盖了线性代数、快速傅里叶变换、连续分布和离散分布以及随机数等内容。

在下一章中，我们将学习一些专用函数。这些函数可能不会经常用到，但当你需要时会非常有用。 

