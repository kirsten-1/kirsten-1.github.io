---
layout: post
title: "MATHEMATICS FOR MACHINE LEARNING第二章 2.1~2.3 阅读笔记"
subtitle: "chap2 Linear Algebra 2.1~2.3"
date: 2025-02-01
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 书籍阅读
---


《MATHEMATICS FOR MACHINE LEARNING》阅读笔记系列：

[chap1 Introduction and Motivation](https://kirsten-1.github.io/2025/01/29/MATHEMATICS-FOR-MACHINE-LEARNING%E7%AC%AC%E4%B8%80%E7%AB%A0%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/)已经整理完。

**本篇整理2.1～2.3节**

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>

# chap2 Linear Algebra - 线性代数

在形式化直觉概念时，常用的方法是构建一组对象`objects`（符号`symbols`）和一组操作这些对象的规则。这被称为**代数(`algebra`)**。**线性代数(`Linear algebra`)**是研究向量以及用于操作向量的某些代数规则的学科。我们许多人从学校学到的向量被称为“**几何向量(`geometric vectors`)**”，通常用字母上方的小箭头表示，例如，$$\vec{x}$$ 和$$\vec{y}$$。在本书中，我们讨论更为一般的向量概念，并使用粗体字母来表示它们，例如，**$$x$$** 和**$$y$$**。

一般来说，向量是特殊的对象，可以相互相加，并可以与标量(`scalars`)相乘以产生同一类型的另一个对象。从抽象数学的角度来看，任何满足这两个性质的对象都可以被视为向量。以下是这类向量对象(`vector objects`)的一些例子：

1. **几何向量(`Geometric vectors`)**。这个向量的例子可能在高中数学和物理中已经很熟悉了。几何向量——见下图——是有方向的线段，可以至少在二维空间中绘制。

   <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250129163037472.png" alt="image-20250129163037472" style="zoom:50%;" />

   两个几何向量$$\vec{x}$$ 和$$\vec{y}$$可以相加，使得$$\vec{x}$$ 和$$\vec{x}+\vec{y}=\vec{z}$$仍然是一个几何向量。此外，标量 $$\lambda$$ 与向量$$\vec{x}$$的乘积$$\lambda\vec{x}$$（其中$$\lambda \in R$$）也是一个几何向量。实际上，它是原向量按比例缩放 λ 倍的结果。因此，几何向量是前面介绍的向量概念的具体实例。将向量解释为几何向量使我们能够利用对方向和大小的直观理解来进行数学运算的推理。

2. **多项式(`Polynomials`)也是向量**。见下图。

   <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250129163132694.png" alt="image-20250129163132694" style="zoom:50%;" />

   两个多项式可以相加，结果得到另一个多项式；它们还可以被一个标量$$\lambda \in R$$乘，结果仍然是一个多项式。因此，多项式是（相当不寻常的）向量实例。需要注意的是，多项式与几何向量非常不同。**几何向量是具体的“图形”，而多项式是抽象的概念。**然而，从前面描述的意义上来说，它们都是向量。

3. 音频信号(`Audio signals`)是向量。音频信号表示为一系列数字。我们可以将音频信号相加，它们的和是一个新的音频信号。如果我们缩放一个音频信号，我们也会得到一个音频信号。因此，音频信号也是一种向量。

4. **$$\mathbb{R}^n$$中的元素（n 个实数的元组）是向量**。$$\mathbb{R}^n$$比多项式更为抽象，也是本书重点关注的概念。例如，
   $$ 
   a = \begin{bmatrix}
   1 \\ 
   2 \\ 
   3  
   \end{bmatrix} \in \mathbb{R}^3 
   $$
   是一个数字三元组的例子。

   将两个向量$$a$$和 $$b \in \mathbb{R}^n$$ 按分量相加，结果是另一个向量：$$a + b = c \in \mathbb{R}^n$$。此外，将$$a \in \mathbb{R}^n$$与$$\lambda \in R$$相乘会得到一个缩放后的向量 $$\lambda a \in \mathbb{R}^n$$。

   将向量视为$$\mathbb{R}^n$$的元素有一个额外的好处，它大致对应于计算机上的实数数组。许多编程语言支持数组操作，这使得涉及向量操作的算法得以方便地实现。

---

线性代数关注这些向量概念之间的相似性(`similarities`)。我们可以将它们相加，并用标量乘以它们。我们将主要关注$$\mathbb{R}^n$$中的向量，因为线性代数中的大多数算法都是在中$$\mathbb{R}^n$$表述的。我们将在第八章看到，我们通常将数据视为$$\mathbb{R}^n$$中的向量表示。在本书中，我们将专注于有限维向量空间，在这种情况下，任何类型的向量与$$\mathbb{R}^n$$之间存在一一对应关系。当方便时，我们将使用几何向量的直观理解，并考虑基于数组的算法。

数学中的一个重要概念是“**闭包(`closure`)**”概念。问题是：从我提议的操作中可以得出的所有事物的集合是什么？对于向量而言：从一小部分向量开始，通过将它们相加和缩放，可以得到的向量集合是什么？这会导致一个**向量空间`a vector space`**（第2.4节）。向量空间的概念及其性质是机器学习的许多基础。本章介绍的概念总结在下图中。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250129165353461.png" alt="image-20250129165353461" style="zoom:50%;" />

线性代数在机器学习和一般数学中起着重要作用。本章引入的概念将在第3章进一步扩展，包括几何思想。在第5章，我们将讨论向量微积分，其中矩阵运算的原理性知识是必不可少的。在第10章，我们将使用投影（将在第3.8节介绍）进行主成分分析（PCA）以实现降维。在第9章，我们将讨论线性回归，其中线性代数在解决最小二乘问题中起着核心作用。

## 2.1 线性方程组-Systems of Linear Equations

线性方程组在线性代数中占据核心地位。许多问题可以表示为线性方程组，而线性代数为我们提供了求解它们的工具。

> **例子2.1**
>
> 一家公司生产产品$$N_1,...,N_n$$，这些产品需要资源$$R_1,...,R_m$$。为了生产一单位的产品$$N_j$$，需要$$a_{ij}$$单位的资源$$R_i$$，其中$$i = 1, ...,m，j = 1, ...,n$$。
>
> 目标是找到一个最优的生产计划，即确定每种产品$$N_j$$应生产多少单位$$x_j$$，使得在总共有$$b_i$$单位的资源$$R_i$$可用的情况下，（理想情况下）不浪费任何资源。如果我们生产 $$x_1，...，x_n$$单位的相应产品，则需要总共需要
> $$
> a_{i1}x_1 + · · · + a_{1n}x_n 
> $$
> 这么多单位的资源 $$R_i$$。一个最优生产计划 $$(x_1, ..., x_n) \in  \mathbb{R}^n$$ 因此必须满足以下方程组：
> \$\$
> a_{11}x_1 + · · · + a_{1n}x_n = b_1 \\
> ... \\
> a_{m1}x_1 + · · · + a_{mn}x_n = b_m
> \$\$
> 其中 $$a_{ij} \in\mathbb{R}$$ 和 $$b_i \in \mathbb{R} $$。

方程$$(3)$$是线性方程组的一般形式，满足$$(3)$$的未知数 $$ (x_1, \ldots, x_n) $$是该方程组的一个解。每一个$$n$$元组$$ (x_1, \ldots, x_n) $$是线性方程组的解。

> **例子2.2**
>
> 线性方程组
> \$\$
> x_1 + x_2 + x_3 = 3 (1)  \\ 
> x_1 − x_2 + 2x_3 = 2 (2)  \\ 
> 2x_1   + 3x_3 = 1 (3)
> \$\$
> 无解：将前两个方程相加得到 $$ 2x_1 + 3x_3 = 5 $$，这与第三个方程$$ (3) $$ 矛盾。
>
> 让我们来看这个线性方程组。
> \$$
> x_1 + x_2 + x_3 = 3 (1)  \\ 
> x_1 − x_2 + 2x_3 = 2 (2)  \\ 
> x_2 + x_3 = 2 (3)
> \$$
> 由第一个和第三个方程可得 $$ x_1 = 1 $$。由$$ (1)+(2) $$可得 $$ 2x_1 + 3x_3 = 5 $$，即 $$ x_3 = 1 $$。由$$(3)$$可得 $$ x_2 = 1 $$。因此，$$ (1,1,1) $$是唯一可能且唯一的解（通过代入验证$$ (1,1,1) $$是解）。
>
> 作为第三个例子，我们考虑
> \$$
> \begin{equation}
> \begin{aligned}
>  x_1 + x_2 + x_3 = 3 (1)  \\ 
>  x_1 − x_2 + 2x_3 = 2 (2)  \\ 
>  2x_1 + 3x_3 = 5 (3)  \\ 
> \$$
> 由于$$ (1)+(2)=(3) $$，我们可以省略第三个方程（冗余）。从$$ (1) $$和$$ (2) $$，我们得到 $$ 2x_1 = 5−3x_3 $$ 和 $$ 2x_2 = 1+x_3 $$。我们将 $$ x_3 = a \in \mathbb{R} $$ 定义为自由变量，使得任何三元组
> \$$
> (\frac{5}{2}-\frac{3}{2}a,\frac{1}{2}+\frac{1}{2}a,a),a \in \mathbb{R}
> \end{aligned}
> \end{equation}
> \$$
> 是线性方程组的解，即我们得到一个包含无限多个解的解集。

一般来说，对于实值线性方程组，我们得到的解要么没有，要么有唯一解，要么有无穷多解。线性回归`Linear regression`（第九章）解决了当我们无法求解线性方程组时的例子2.1的版本。

备注（线性方程组的几何解释）。在一个包含两个变量 $$ x_1 $$ 和 $$ x_2 $$的线性方程组中，每个线性方程在 $$ x_1x_2 $$ 平面上定义了一条直线。由于线性方程组的解必须同时满足所有方程，**因此解集是这些直线的交点。**这个交集可以是一条直线（如果线性方程描述的是同一条直线），一个点，或者为空（当直线平行时）。下图给出了下面线性方程组的示例。
\$$
4x_1 + 4x_2 = 5  \\ 
2x_1 − 4x_2 = 1
\$$
<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250131210518541.png" alt="image-20250131210518541" style="zoom:50%;" />

其中解空间是点 $$ (x_1, x_2) = (1, \frac{1}{4}) $$。类似地，对于三个变量，每个线性方程确定了三维空间中的一个平面。当我们求这些平面的交集，即同时满足所有线性方程时，我们可以得到一个解集，这个解集可以是一个平面、一条直线、一个点或为空（当这些平面没有共同交点时）。

为了解线性方程组，我们将介绍一种有用的紧凑符号。我们将系数 $$ a_{ij} $$ 收集到向量中，再将这些向量收集到矩阵中。换句话说，我们将 $$ (3) $$ 中的方程组写成以下形式：
\$$
\begin{bmatrix} 
a_{11}  \\  
... \\
a_{m1}  
\end{bmatrix} x_1 + 
\begin{bmatrix} 
a_{12} 
\\ 
... \\ 
a_{m2}  
\end{bmatrix} x_2 + ... + 
\begin{bmatrix} a_{1n} \\
... \\ 
a_{mn}  
\end{bmatrix} x_n = 
\begin{bmatrix} b_{1} \\
... \\ 
b_{m}  
\end{bmatrix}
\Leftrightarrow 
\begin{bmatrix} 
a_{11} ... a_{1n} \\ 
... \\ 
a_{m1}...a_{mn}  
\end{bmatrix} 
\begin{bmatrix} x_{1} \\ 
... \\ 
x_{n}  
\end{bmatrix} = 
\begin{bmatrix} b_{1} \\
... \\ 
b_{m}  
\end{bmatrix}
\$$
接下来，我们将详细研究这些矩阵并定义计算规则。我们将在2.3节回到求解线性方程的问题。

## 2.2 矩阵-Matrices

矩阵在线性代数中起着核心作用。它们可以用来紧凑地表示线性方程组，同时也能表示线性函数（线性映射），我们将在2.7节中看到。在讨论这些有趣的话题之前，让我们先定义什么是矩阵以及我们可以对矩阵进行哪些操作。我们将在第4章中看到更多关于矩阵的性质。

**定义2.1**矩阵`Matrix`。对于 $$m, n \in \mathbb{N}$$，一个实值 $$(m, n)$$ 矩阵 **$$A$$** 是一个 $$m·n$$ 元组，其元素为 $$a_{ij}$$，其中 $$i = 1, . . . , m，j = 1, . . . , n$$，这些元素按照由 m 行和 n 列组成的矩形方案排列：
\$$
A=\begin{bmatrix}a_{11} \ a_{12}\ ...\ a_{1n} \\a_{21}\ a_{22}\ ...\ a_{2n}\\ . \ . \ . \\\ a_{m1}\ a_{m2}\ ...\ a_{mn} \end{bmatrix},a_{ij}\in \mathbb{R}
\$$
按照惯例，$$(1, n)$$矩阵被称为行，$$(m, 1)$$矩阵被称为列。这些特殊的矩阵也被称为**行向量/列向量**。

$$\mathbb{R}^{m×n}$$ 是所有实值 $$(m, n)$$ 矩阵的集合。$$A \in \mathbb{R}^{m×n}$$ 可以等价地表示为$ a \in  \mathbb{R}^{mn}$，通过将矩阵的所有 n 列堆叠成一个长向量；见下图。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250131212354250.png" alt="image-20250131212354250" style="zoom:50%;" />

### 2.2.1 矩阵加法和乘法-Matrix Addition and Multiplication

两个矩阵 $$A \in \mathbb{R}^{m×n}$$，$$B \in \mathbb{R}^{m×n}$$的和定义为**元素级的和**，即
\$$
A+B=\begin{bmatrix}a_{11}+b_{11} \ \  \ ...\ \  a_{1n}+b_{1n}\\ . \ . \ . \\\ a_{m1}+b_{m1}\ \  \ ...\ \ a_{mn}+b_{mn} \end{bmatrix}\in \mathbb{R}^{m×n}
\$$
对于矩阵$$A \in \mathbb{R}^{m×n}$$，$$B \in \mathbb{R}^{n×k}$$，两者乘积$$C=AB\in \mathbb{R}^{m×k}$$的元素 \( $$c_{ij}$$ \) 计算为
\$\$
c_{ij}=\sum_{l=1}^n a_{il}b_{lj},i=1,...,m,j=1,...,k
\$\$
这意味着，为了计算元素 $$c_{ij}$$，我们将 A 的第 i 行与 B 的第 j 列的元素相乘并求和。在稍后的 3.2 节中，我们将这种对应行和列的乘积和称为**点积(`dot product`)**。在需要明确表示乘法的情况下，我们使用符号 `A · B` 来表示乘法（明确显示“`·`”）。

> 由于矩阵 A有 n 列，矩阵 B有 n 行，因此我们可以计算 $$a_{il} b_{lj}$$，其中 $$l=1,…,n$$。通常，两个向量 $$a$$ 和 $$b$$ 之间的点积表示为 $$a^T b$$ 或$$\langle a, b \rangle$$。

**备注：** 只有当矩阵的“相邻”维度匹配时，矩阵才能相乘。例如，一个 $$n \times k$$的矩阵 $$A$$ 可以与一个 $$k \times m$$ 的矩阵 $$B$$ 进行乘法运算，但只能从左边进行：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250201125939091.png" alt="image-20250201125939091" style="zoom:50%;" />

如果 $$m \neq n$$，则 $$BA$$ 的乘积是未定义的，因为相邻的维度不匹配。

**备注：** 矩阵乘法不是矩阵元素逐一运算的操作，也就是说，$$c_{ij} \neq a_{ij} b_{ij}$$（即使矩阵 $$A$$ 和 $$B$$ 的大小选择得当）。这种逐元素的乘法通常出现在编程语言中，当我们将（多维）数组相乘时，称之为 **Hadamard 乘积(`Hadamard product`)**。

> 矩阵乘法是一种更复杂的操作，它基于点积原理，而Hadamard乘积是一个元素级的操作，两者不应混淆。

> **例子2.3**
>
> 对于$$A = \begin{bmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3} $$,$$B = \begin{bmatrix} 0 & 2 \\ 1 & -1 \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$,可以得到：
> \$\$
> AB = \begin{bmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \end{bmatrix} \begin{bmatrix} 0 & 2 \\ 1 & -1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 3 \\ 2 & 5 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
> \$\$
>
> \$$
> BA = \begin{bmatrix} 0 & 2 \\ 1 & -1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 6 & 4 & 2 \\ -2 & 0 & 2 \\ 3 & 2 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 3}
> \$$

从这个例子中，我们已经可以看到矩阵乘法是**不可交换**的，即 $$AB \neq BA$$；另见下图以获得图示说明。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250201130835457.png" alt="image-20250201130835457" style="zoom:50%;" />

**定义2.2 单位矩阵`Identity Matrix`**：在 $$\mathbb{R}^{n \times n} $$中，我们定义单位矩阵 \( $$I_n$$ \) 为一个 \( $$n \times n$$ \) 的矩阵，主对角线上的元素为 1，其他位置的元素为 0。
\$\$
I_n := \begin{bmatrix}
1 & 0 & \dots & 0 & \dots & 0 \\
0 & 1 & \dots & 0 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 1 & \dots & 0 \\
0 & 0 & \dots & 0 & \dots & 1 \\
\end{bmatrix} \in \mathbb{R}^{n \times n}
\$\$
**矩阵的性质**

现在我们已经定义了矩阵乘法、矩阵加法以及单位矩阵，让我们看看矩阵的一些基本性质：

**结合性`Associativity`**:对于  $$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}, C \in \mathbb{R}^{p \times q} : (AB)C = A(BC)$$

**分配性`Distributivity`**:对于$$A, B \in \mathbb{R}^{m \times n}, C, D \in \mathbb{R}^{n \times p}$$:
\$$
(A + B)C = AC + BC
\$$

\$$
A(C + D) = AC + AD
\$$

**与单位矩阵的乘法**:对于$$A \in \mathbb{R}^{m \times n}$$:$$I_m A = A I_n = A$$

注意，对于 $$m \neq n $$，有 $$ I_m \neq I_n $$。

### 2.2.2 逆和转置-Inverse and Transpose

**定义 2.3 (逆矩阵)`Inverse`**:考虑一个方阵 $$ A \in \mathbb{R}^{n \times n} $$，若矩阵 $$ B $$ 满足 $$ AB = I_n = BA $$，则称 $$ B $$ 为 $$ A $$ 的逆矩阵，记作 $$ A^{-1} $$。

不幸的是，并不是每个矩阵 $$ A $$ 都存在逆矩阵 $$ A^{-1} $$。如果矩阵的逆存在，则称 $$ A $$ 为正则/可逆/非奇异矩阵（`regular/invertible/nonsingular`），否则称为奇异矩阵（`singular/noninvertible`）。当矩阵的逆存在时，它是唯一的。在第 3 节中，我们将讨论一种通过解线性方程组来计算矩阵逆的方法。

**备注（2×2矩阵的逆存在性）**：考虑矩阵  $$ A := \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \in \mathbb{R}^{2 \times 2} $$

如果我们用矩阵 $$ A $$ 乘以矩阵  $$ A' := \begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix} $$

我们得到  $$ AA' = \begin{bmatrix} a_{11}a_{22} - a_{12}a_{21} & 0 \\ 0 & a_{11}a_{22} - a_{12}a_{21} \end{bmatrix} = (a_{11}a_{22} - a_{12}a_{21})I $$

因此，  $$ A^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}} \begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix} $$

当且仅当 $$ a_{11}a_{22} - a_{12}a_{21} \neq 0 $$ 时，矩阵 $$ A $$ 可逆。在第 4.1 节中，我们将看到 $$ a_{11}a_{22} - a_{12}a_{21} $$ 是 2×2 矩阵的行列式。此外，我们通常可以使用行列式来检查矩阵是否可逆。

> **例子 2.4 (逆矩阵)**
>
> 矩阵  $$ A = \begin{bmatrix} 1 & 2 & 1 \\ 4 & 4 & 5 \\ 6 & 7 & 7 \end{bmatrix} $$  与  $$ B = \begin{bmatrix} -7 & -7 & 6 \\ 2 & 1 & -1 \\ 4 & 5 & -4 \end{bmatrix} $$  是互为逆矩阵，因为  $$ AB = I = BA $$。

**定义 2.4 (转置)`Transpose`**对于 $$ A \in \mathbb{R}^{m \times n} $$，矩阵 $$ B \in \mathbb{R}^{n \times m} $$，其中 $$ b_{ij} = a_{ji} $$，称 $$ B $$ 为 $$ A $$ 的转置，记作 $$ B = A^T $$。

通常，$$ A^T $$ 可以通过将矩阵 $$ A $$ 的列作为行来得到 $$ A^T $$。以下是逆矩阵和转置矩阵的一些重要性质：

\$\$
AA^{-1} = I = A^{-1}A\\
(AB)^{-1} = B^{-1}A^{-1}\\
(A + B)^{-1} \neq A^{-1} + B^{-1}\\
(A^T)^T = A\\
(AB)^T = B^T A^T\\
(A + B)^T = A^T + B^T
\$\$
**定义 2.5 (对称矩阵)`Symmetric Matrix`**:矩阵 $$ A \in \mathbb{R}^{n \times n} $$ 是对称矩阵，当且仅当 $$ A = A^T $$。

注意，只有 $$ (n, n) $$ 矩阵才能是对称的。通常我们称 $$ (n, n) $$ 矩阵为方阵，因为它们具有相同的行数和列数。此外，如果矩阵 $$ A $$ 可逆，则 $$ A^T $$ 也可逆，并且有  
$$ (A^{-1})^T = (A^T)^{-1} = A^{-T} $$。

**备注（对称矩阵的和与积）**：对称矩阵 $$ A, B \in \mathbb{R}^{n \times n} $$ 的和总是对称的。然而，尽管它们的乘积总是定义的(可以计算的)，但通常情况下它们的积并不是对称的：
\$\$
\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}
\$\$

### 2.2.3 标量乘法-Multiplication by a Scalar

让我们看看当矩阵与标量 $$ \lambda \in \mathbb{R} $$ 相乘时会发生什么。设 $$ A \in \mathbb{R}^{m \times n} $$ 且 $$ \lambda \in \mathbb{R} $$，则有   $$ \lambda A = K, \ K_{ij} = \lambda a_{ij} $$。   实际上，$$ \lambda $$ 会缩放矩阵 $$ A $$ 中的每个元素。对于 $$ \lambda, \psi \in \mathbb{R} $$，以下关系成立：

- **结合性`Associativity`**：    $$ (\lambda \psi)C = \lambda (\psi C), \ C \in \mathbb{R}^{m \times n} $$
- $$ \lambda (BC) = ( \lambda B ) C = B( \lambda C ) = (BC)\lambda, \ B \in \mathbb{R}^{m \times n}, C \in \mathbb{R}^{n \times k} $$    注意，这允许我们在标量值之间移动。

- **转置的性质**：    $$ (\lambda C)^T = C^T \lambda^T = C^T \lambda = \lambda C^T \quad \text{因为}\text{对于所有} \ \lambda \in \mathbb{R} \ ,有 \lambda = \lambda^T \ . $$
- **分配性`Distributivity`**：    $$ (\lambda + \psi)C = \lambda C + \psi C, \ C \in \mathbb{R}^{m \times n} ；$$    $$ \lambda (B + C) = \lambda B + \lambda C, \ B, C \in \mathbb{R}^{m \times n} $$

> **例子2.5 (分配性)**
>
> 如果我们定义   $$ C := \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad  $$   那么对于任意 $$ \lambda, \psi \in \mathbb{R} $$，我们有： $$ (\lambda + \psi)C = \begin{bmatrix} (\lambda + \psi)1 & (\lambda + \psi)2 \\ (\lambda + \psi)3 & (\lambda + \psi)4 \end{bmatrix} = \begin{bmatrix} \lambda + \psi & 2\lambda + 2\psi \\ 3\lambda + 3\psi & 4\lambda + 4\psi \end{bmatrix} = \begin{bmatrix} \lambda & 2\lambda \\ 3\lambda & 4\lambda \end{bmatrix} + \begin{bmatrix} \psi & 2\psi \\ 3\psi & 4\psi \end{bmatrix} = \lambda C + \psi C \quad$$

### 2.2.4 线性方程组的紧凑表示-Compact Representations of Systems of Linear Equations

假设我们考虑如下的线性方程组：

\$\$
2x_1 + 3x_2 + 5x_3 = 1\\
4x_1 - 2x_2 - 7x_3 = 8 \\
9x_1 + 5x_2 - 3x_3 = 2
\$\$
通过使用矩阵乘法的规则，我们可以将此方程组写成更紧凑的形式，如下所示：

\$$
\begin{bmatrix}
2 & 3 & 5 \\
4 & -2 & -7 \\
9 & 5 & -3
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
=
\begin{bmatrix}
1 \\
8 \\
2
\end{bmatrix}
\$$
注意，$$ x_1 $$ 缩放了第一列，$$ x_2 $$ 缩放了第二列，$$ x_3 $$ 缩放了第三列。

一般来说，线性方程组可以紧凑地表示为矩阵形式 $$ A x = b $$；参见(2.3)，其中 $$ A x $$ 是矩阵 $$ A $$ 列的线性组合` linear combinations`。我们将在第 2.5 节中更详细地讨论线性组合。

## 2.3 解线性方程组-Solving Systems of Linear Equations

在公式 $$(3) $$中，我们介绍了一个方程组的通用形式，即：
\$\$
a_{11}x_1 + · · · + a_{1n}x_n = b_1\\
...\\
a_{m1}x_1 + · · · + a_{mn}x_n = b_m
\$\$
其中 $$ a_{ij} \in \mathbb{R} $$ 和 $$ b_i \in \mathbb{R} $$ 是已知常数，$$ x_j $$ 是未知数，$$ i = 1, \dots, m, j = 1, \dots, n $$。到目前为止，我们已经看到矩阵可以用来紧凑地表示线性方程组，从而我们可以写出 $$ A x = b $$，见公式 $$(9)$$。此外，我们还定义了基本的矩阵操作，如矩阵的加法和乘法。在接下来的内容中，我们将专注于求解线性方程组并提供一个算法来计算矩阵的逆。

### 2.3.1 特解和通解-Particular and General Solution

在讨论如何一般性地求解线性方程组之前，我们先看一个例子。考虑如下方程组：
\$$
\begin{bmatrix} 1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix}
=
\begin{bmatrix} 42 \\ 8 \end{bmatrix}
\$$
该方程组有两个方程和四个未知数。因此，一般来说，我们预期会有无穷多解。这个方程组的形式特别简单，前两列是一个 1 和一个 0。记住，我们想要找到标量 $$ x_1, \dots, x_4 $$，使得$$\sum_{i=1}^4 x_i c_i = b,$$其中我们定义 $$ c_i $$ 为矩阵的第 $$ i $$ 列，而 $$ b $$ 是方程组右边的值。我们可以通过以下方式求解 $$(23)$$ 中的问题：取 42 倍的第一列和 8 倍的第二列：
\$\$
b = \begin{bmatrix} 42 \\ 8 \end{bmatrix} = 42 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 8 \begin{bmatrix} 0 \\ 1 \end{bmatrix} \
\$\$
因此，一个解是 $$ [42, 8, 0, 0]^T $$。这个解被称为**特解（particular solution）**。然而，这并不是这个方程组唯一的解。为了捕捉所有解，我们需要通过矩阵的列来创造一个非平凡的 0，使用如下方式：添加 0 到特解中并不会改变解的特性。为此，我们用前两列表示第三列，以生成非常简单的 0：
\$$
\begin{bmatrix} 8 \\ 2 \end{bmatrix} = 8 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 2 \begin{bmatrix} 0 \\ 1 \end{bmatrix}
\$$
所以 $$ 0 = 8 c_1 + 2 c_2 - 1 c_3 + 0 c_4 $$，并且 $$ (x_1, x_2, x_3, x_4) = (8, 2, -1, 0) $$。 实际上，任何 $$ \lambda_1 \in \mathbb{R} $$ 对这个解的缩放都会生成 0 向量，即：
\$\$
\begin{bmatrix} 1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix}
\begin{bmatrix} \lambda_1 \end{bmatrix}
= \lambda_1 \begin{bmatrix} 8 c_1 + 2 c_2 - c_3 \end{bmatrix} = 0
\$\$
同样的推理，我们用前两列表示矩阵的第四列，生成另一组非平凡的 0：
\$$
\begin{bmatrix} 1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix}
\begin{bmatrix} \lambda_2 \end{bmatrix}
= \lambda_2 \begin{bmatrix} -4c_1 + 12c_2 - c_4 \end{bmatrix} = 0
\$$
将所有内容结合起来，我们得到方程组 $$(23)$$ 的所有解，称为**通解（general solution）**：
\$\$
\begin{bmatrix} x \in \mathbb{R}^4 : x = \begin{bmatrix} 42 \\ 8 \\ 0 \\ 0 \end{bmatrix} + \lambda_1 \begin{bmatrix} 8 \\ 2 \\ -1 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} -4 \\ 12 \\ 0 \\ -1 \end{bmatrix}, \lambda_1, \lambda_2 \in \mathbb{R} \end{bmatrix}
\$\$
备注:我们遵循的一般方法包括以下三个步骤：

1. 找到 $$ Ax = b $$ 的一个特解。
2. 找到所有 $$ Ax = 0 $$ 的解。
3. 将步骤 1 和步骤 2 的解结合起来得到通解。

通解和特解都不是唯一的。 前面的例子中的线性方程组容易求解，因为矩阵 $$(23)$$ 具有特别方便的形式，这使我们能够通过观察找到特解和通解。然而，一般的方程组并不具有这么简单的形式。幸运的是，存在一种构造性的算法，通过高斯消元法将任何线性方程组转化为这种特别简单的形式。高斯消元法的关键是矩阵的基本变换，它们将方程组转化为更简单的形式。然后，我们可以将前三个步骤应用到这种简单形式的方程组中，正如我们在 $$(23)$$ 中所做的那样。

> 上面的说实话  我也不是很想理解，但是以前也学习过（本科+考研都学过），参考[视频](https://www.bilibili.com/video/BV1aW411Q7x1?spm_id_from=333.788.videopod.episodes&vd_source=6c6e2754e61f483e81b4bc03c9898c87&p=43)（找的一个通俗易懂的视频）
>
> 截图如下：
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250201141748675.png" alt="image-20250201141748675" style="zoom:50%;" />
>
> <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250201142145260.png" alt="image-20250201142145260" style="zoom:50%;" />
>
> 举例：
>
> ![image-20250201143444850](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250201143444850.png)

### 2.3.2 初等变换-Elementary Transformations

解决线性方程组的关键是**初等变换**，这些变换保持解集不变，但将方程组转换为更简单的形式：

- 交换两个方程（矩阵中代表方程的行）
- 用常数 $$ \lambda \in \mathbb{R} \setminus \{0\} $$ 乘以一个方程（行）
- 相加两个方程（行）

> **例子2.6**
>
> 对于 $$ a \in \mathbb{R} $$，我们寻求如下方程组的所有解：
> \$\$
> -2x_1 + 4x_2 - 2x_3 - x_4 + 4x_5 = -3\\
> 4x_1 - 8x_2 + 3x_3 - 3x_4 + x_5 = 2\\
> x_1 - 2x_2 + x_3 - x_4 + x_5 = 0\\
> x_1 - 2x_2 - 3x_4 + 4x_5 = a
> \$\$
> 我们开始将这个方程组转换为紧凑的矩阵表示 $$ Ax = b $$。我们不再显式提及变量 $$ x $$，而是构建增广矩阵`augmented matrix` $$ [A|b] $$，形式如下：
> \$$
> \begin{bmatrix}
> -2 & 4 & -2 & -1 & 4 & -3 \\
> 4 & -8 & 3 & -3 & 1 & 2 \\
> 1 & -2 & 1 & -1 & 1 & 0 \\
> 1 & -2 & 0 & -3 & 4 & a
> \end{bmatrix}
> \$$
> 我们用竖线分隔矩阵的左边和右边，并使用 $$ \Rightarrow $$ 表示增广矩阵的变换。交换行 1 和行 3 后得到：
> \$\$
> \begin{bmatrix}
> 1 & -2 & 1 & -1 & 1 & 0 \\
> 4 & -8 & 3 & -3 & 1 & 2 \\
> -2 & 4 & -2 & -1 & 4 & -3 \\
> 1 & -2 & 0 & -3 & 4 & a
> \end{bmatrix}
> \$\$
> 现在应用所指示的变换（例如，从第二行减去四倍的第一行），3，4行也做类似的变换，得到：
> \$$
> \begin{bmatrix}
> 1 & -2 & 1 & -1 & 1 & 0 \\
> 0 & 0 & -1 & 1 & -3 & 2 \\
> 0 & 0 & 0 & -3 & 6 & -3 \\
> 0 & 0 & -1 & -2 & 3 & a
> \end{bmatrix}
> \$$
> 接下来应用第二次变换，得到：
> \$\$
> \begin{bmatrix}
> 1 & -2 & 1 & -1 & 1 & 0 \\
> 0 & 0 & -1 & 1 & -3 & 2 \\
> 0 & 0 & 0 & -3 & 6 & -3 \\
> 0 & 0 & 0 & -3 & 6 & a-2
> \end{bmatrix} \Longrightarrow
> \begin{bmatrix}
> 1 & -2 & 1 & -1 & 1 & 0 \\
> 0 & 0 & -1 & 1 & -3 & 2 \\
> 0 & 0 & 0 & -3 & 6 & -3 \\
> 0 & 0 & 0 & 0 & 0 & a+1
> \end{bmatrix}\Longrightarrow
> \begin{bmatrix}
> 1 & -2 & 1 & -1 & 1 & 0 \\
> 0 & 0 & 1 & -1 & 3 & -2 \\
> 0 & 0 & 0 & 1 & -2 & 1 \\
> 0 & 0 & 0 & 0 & 0 & a+1
> \end{bmatrix}
> \$$
> 这个增广矩阵现在是一个简洁的行阶梯形式（REF,`row-echelon form`）。将这个简化的矩阵重新表示为我们寻求的变量：
> \$\$
> x_1 - 2x_2 + x_3 - x_4 + x_5 = 0\\
> x_3 - x_4 + 3x_5 = -2\\
> x_4 - 2x_5 = 1\\
> 0 = a + 1
> \$\$
> 只有当 $$ a = -1 $$ 时，系统才能求解。特解是：
> \$$
> \begin{bmatrix}
> x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5
> \end{bmatrix}=\begin{bmatrix}
> 2 \\ 0 \\ -1 \\ 1 \\ 0
> \end{bmatrix}
> \$$
> 通解，捕捉所有可能解的集合是：
> \$$
> x \in \mathbb{R}^5 : x =
> \begin{bmatrix}
> 2 \\ 0 \\ -1 \\ 1 \\ 0
> \end{bmatrix} + \lambda_1
> \begin{bmatrix}
> 2 \\ 1 \\ 0 \\ 0 \\ 0
> \end{bmatrix} + \lambda_2
> \begin{bmatrix}
> 2 \\ 0 \\ -1 \\ 2 \\ 1
> \end{bmatrix}, \lambda_1, \lambda_2 \in \mathbb{R}
> \$$

备注:我们遵循的通用方法包含以下三步：

1. 找到 $$ Ax = b $$ 的一个特解。
2. 找到所有 $$ Ax = 0 $$ 的解。
3. 将步骤 1 和步骤 2 的解结合起来得到通解。

特解和通解都不是唯一的。 通过高斯消元法，任何线性方程组都可以转换成简化形式。高斯消元法的关键是矩阵的初等变换，它们将方程系统转换为更简单的形式。我们将在下面详细介绍如何获得一个线性方程组的特解和通解。

----

**定义 2.6 (行阶梯形`Row-Echelon Form`)** 一个矩阵处于（row-echelon）形式，当且仅当满足以下条件：

- 所有仅包含零的行位于矩阵的底部；与之相对，所有包含至少一个非零元素的行都位于仅包含零的行之上。
- 只考虑非零行时，来自左侧的第一个非零数（也称为主元或首项）始终严格位于其上一行主元的右侧。

> 关于行阶梯形和简化行阶梯形，可参考[帖子](https://zhuanlan.zhihu.com/p/86493143)

备注:（基本变量与自由变量） 在行阶梯形中，与主元对应的变量称为基本变量，而其他变量称为自由变量。例如，在 (34) 中，$$ x_1, x_3, x_4 $$ 是基本变量，而 $$ x_2, x_5 $$ 是自由变量。

备注（求解特解） 行阶梯形使得我们在需要求解特解时能更轻松地操作。为了求解特解，我们用主元列表示方程组的右边，形如 $$ b = \sum_{i=1}^P \lambda_i p_i $$，其中 $$ p_i $$ 是主元列，$$ \lambda_i $$ 是待求解的标量，按照从右到左的顺序逐步确定它们。 在前一个例子中，我们会尝试找到 $$ \lambda_1, \lambda_2, \lambda_3 $$，使得：
\$$
\begin{bmatrix} 1 \\ 0 \\ 0 \\0\end{bmatrix} \lambda_1 +
\begin{bmatrix} 1 \\1 \\ 0 \\ 0 \end{bmatrix} \lambda_2 +
\begin{bmatrix} -1 \\ -1 \\ 1 \\ 0 \end{bmatrix} \lambda_3 =
\begin{bmatrix} 0 \\ -2 \\ 1 \\0 \end{bmatrix}
\$$
由此，我们可以直接得到 $$ \lambda_3 = 1 $$，$$ \lambda_2 = -1 $$，$$ \lambda_1 = 2 $$。当我们把所有值代入时，不要忘记那些非主元列，它们的系数隐含为零。因此，我们得到特解：$$x = \begin{bmatrix} 2 , 0 , -1 ,1, 0 \end{bmatrix}^T $$

备注（简化行阶梯形） 简化行阶梯形在 2.3.3 节中会发挥重要作用，因为它能帮助我们直接得到线性方程组的通解。

备注（高斯消元法） 高斯消元法是一种将线性方程组转换为简化行阶梯形的算法。

> **例子2.7 (简化行阶梯形)**
>
> 验证下列矩阵是否为简化行阶梯形（主元为粗体）：
> \$$
> A = \begin{bmatrix} 1 & 3 & 0 & 0 & 3 \\ 0 & 0 & 1 & 0 & 9 \\ 0 & 0 & 0 & 1 & -4 \end{bmatrix}
> \$$
> 求解 $$ Ax = 0 $$ 的关键思想是观察非主元列，并将它们表示为主元列的线性组合。简化行阶梯形使得这一过程非常简单，我们用主元列和它们左侧的非主元列表示其他列。我们需要从第一列中减去第三列三倍，再处理第五列，这就是第二个非主元列。 因此，我们得到：
> \$$
> x \in \mathbb{R}^5 : x = \lambda_1 \begin{bmatrix} 3 \\ -1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\ 0 \\ 9 \\ -4 \\ -1 \end{bmatrix} \quad \lambda_1, \lambda_2 \in \mathbb{R}
> \$$
> 补充：令$$\begin{bmatrix}x_2\\x_5 \end{bmatrix} = \begin{bmatrix} 1\\0 \end{bmatrix}$$和$$\begin{bmatrix} 0\\1 \end{bmatrix}$$,	由$$\begin{cases} x_1= - 3x_2 - 3x_5  \\ x_3= -9x_5  \\ x_4=4x_5 \end{cases}$$,得到$$\eta_1=\begin{bmatrix}-3\\1\\0\\0\\0 \end{bmatrix},\eta_2=\begin{bmatrix}-3\\0\\-9\\4\\1 \end{bmatrix}$$.
>
> 和上面的只差了一个符号。

### 2.3.3 负一技巧-The Minus-1 Trick

> 这个技巧我也是第一次看见，如果早点学到可以更加快速的解考研题目或者应用在其他地方吧。真的是很奇妙

在下面的内容中，我们介绍了一种实用的技巧，用于求解齐次线性方程组 $Ax = 0$ 的解，其中 $A \in \mathbb{R}^{k \times n}$，$x \in \mathbb{R}$。
首先，我们假设矩阵 $A$ 已经是**简化行最简形式（reduced row-echelon form）**，且没有任何只包含零的行，即：
\$$
A =
\begin{bmatrix}
0 & \cdots & 0 & 1 & \cdots & * & 0 & * & \cdots & * \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & \cdots & 0 & 0 & \cdots & 1 & * & \cdots & * \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & \cdots & 0 & 0 & \cdots & 0 & 1 & * & \cdots & * \\
\end{bmatrix}
\$$

其中 $*$ 可以是任意实数，约定每行的第一个非零元素必须是 1，且该列的其他元素必须为 0。标记为粗体的列 $j_1, \dots, j_k$ 是枢纽列（pivots），这些列是标准的单位向量 $e_1, \dots, e_k \in \mathbb{R}^k$。我们通过在矩阵中添加 $n - k$ 行，扩展这个矩阵到一个 $n \times n$ 的矩阵 $\tilde{A}$，这些行的形式为

\$$
\begin{bmatrix}
0 & \cdots & 0 & -1 & 0 & \cdots & 0
\end{bmatrix}
\$$

这样增广矩阵 $\tilde{A}$ 的对角线元素就包含了 1 或 -1。然后，$\tilde{A}$ 中对角线上包含 -1 的列即为齐次方程 $Ax = 0$ 的解。这些列可以构成齐次方程解空间的基（参见 2.6.1 节），我们将其称为核空间或零空间`kernel or null space`（详见 2.7.3 节）。

> **例子 2.8（Minus-1 Trick）**
>
> 我们重新回顾 (38) 中的矩阵，它已经是简化的行最简形式（REF）：
>
> \$$
> A = \begin{bmatrix} 1 & 3 & 0 & 0 & 3 \\ 0 & 0 & 1 & 0 & 9 \\ 0 & 0 & 0 & 1 & -4 \end{bmatrix}
> \$$
>
> 我们现在通过在对角线缺失枢纽的位置添加形式为 (41) 的行，将这个矩阵扩展为一个 $5 \times 5$ 的矩阵，得到
>
> \$$
> \tilde{A} = \begin{bmatrix}
> 1 & 3 & 0 & 0 & 3 \\
> 0 & -1 & 0 & 0 & 0 \\
> 0 & 0 & 1 & 0 & 9 \\
> 0 & 0 & 0 & 1 & -4  \\
> 0 & 0 & 0 & 0 & -1 \\
> \end{bmatrix}
> \$$
>
> 通过这种方式，我们可以立即从矩阵 $\tilde{A}$ 中读取出齐次方程 $Ax = 0$ 的解，方法是取出矩阵中对角线含有 -1 的列：
>
> \$$
> x \in \mathbb{R}^5 : x = \lambda_1 \begin{bmatrix} 3 \\ -1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\ 0 \\ 9 \\ -4 \\ -1 \end{bmatrix}, \lambda_1, \lambda_2 \in \mathbb{R}
> \$$
>
> 这与我们通过“直觉”得到的解 (39) 是一致的。

----

#### **计算逆矩阵**-Calculating the Inverse

为了计算 $A \in \mathbb{R}^{n \times n}$ 的逆矩阵 $A^{-1}$，我们需要找到一个矩阵 $X$，使得满足 $AX = I_n$。然后，$X = A^{-1}$。我们可以将其写成一组线性方程 $AX = I_n$，在其中解得 $X = [x_1 | \cdots | x_n]$。我们使用增广矩阵的符号来简洁地表示这一组线性方程组，得到：

$$
[A | I_n] \xrightarrow{\text{变换}} \cdots \xrightarrow{\text{变换}} [I_n | A^{-1}] \tag{2.56}
$$

这意味着，如果我们将增广方程组变换为简化行最简形式，那么可以从方程组的右侧读取出逆矩阵。由此可见，计算矩阵的逆矩阵等价于求解线性方程组。

> **例子 2.9（通过高斯消元法计算逆矩阵）**
>
> 为了确定矩阵
>
> \$$
> A = \begin{bmatrix}
> 1 & 0 & 2 & 0 \\
> 1 & 1 & 0 & 0 \\
> 1 & 2 & 0 & 1 \\
> 1 & 1 & 1 & 1
> \end{bmatrix}
> \tag{2.57}
> \$$
>
> 的逆矩阵，我们写出增广矩阵：
>
> \$$
> \left[
> \begin{array}{cccc|cccc}
> 1 & 0 & 2 & 0 & 1 & 0 & 0 & 0 \\
> 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
> 1 & 2 & 0 & 1 & 0 & 0 & 1 & 0 \\
> 1 & 1 & 1 & 1 & 0 & 0 & 0 & 1
> \end{array}
> \right]
> \$$
>
> 并使用高斯消元法将其转换为简化行最简形式：
>
> \$\$
> \left[
> \begin{array}{cccc|cccc}
> 1 & 0 & 0 & 0 & -1 & 2 & -2 & 2 \\
> 0 & 1 & 0 & 0 & 1 & -1 & 2 & -2 \\
> 0 & 0 & 1 & 0 & -1 & 1 & 1 & -1 \\
> 0 & 0 & 0 & 1 & -1 & 0 & -1 & 2
> \end{array}
> \right]
> \$\$
>
> 因此，所需的逆矩阵为右侧的部分：
>
> \$$
> A^{-1} = \begin{bmatrix}
> -1 & 2 & -2 & 2 \\
> 1 & -1 & 2 & -2 \\
> 1 & -1 & 1 & -1 \\
> -1 & 0 & -1 & 2
> \end{bmatrix}
> \tag{2.58}
> \$$
>
> 我们可以通过执行矩阵乘法 $AA^{-1}$ 来验证 (2.58) 确实是逆矩阵，并且观察到我们恢复了单位矩阵 $I_4$。





### 2.3.4 解线性方程组的算法-Algorithms for Solving a System of Linear Equations

在下面的内容中，我们简要讨论了解决形如 $Ax = b$ 的线性方程组的方法。我们假设存在解。如果没有解，我们需要转向近似解，本章不涉及此类内容。解决近似问题的一种方法是使用**线性回归**方法，我们将在第9章详细讨论。

在特殊情况下，我们可能能够确定逆矩阵 $A^{-1}$，使得 $Ax = b$ 的解可以表示为 $x = A^{-1}b$。然而，只有当 $A$ 是一个方阵并且是可逆的时候，这种情况才成立，这在实际中并不常见。否则，在一些温和假设（即，$A$ 需要有线性无关的列）的情况下，我们可以使用变换：

\$$
Ax = b \iff A^\top A x = A^\top b \iff x = (A^\top A)^{-1} A^\top b \tag{2.59}
\$$

并使用**摩尔-彭罗斯伪逆（$A^\top A)^{-1} A^\top$）**来确定解 (2.59)，它解出了方程 $Ax = b$，并且也是最小二乘法解(即最小化$$||Ax−b||^2$$)的解的对应方式。这个方法的一个缺点是，它需要计算矩阵乘积和 $(A^\top A)$ 的逆，这涉及到大量计算。此外，由于数值精度的原因，一般不推荐计算矩阵的逆或伪逆。因此，下面我们简要讨论了其他求解线性方程组的方法。

高斯消元法在计算行列式（第4.1节）、检查向量组是否线性无关（第2.5节）、计算矩阵的逆（第2.2节）、计算矩阵的秩（第2.6.2节）以及确定向量空间基（第2.6.1节）等方面起着重要作用。高斯消元法是一种直观且构造性的方法，用于求解成千上万变量的线性方程组。然而，对于成千上万变量的方程组，由于需要的算术运算量呈立方增长，它在计算上变得不切实际。

在实践中，许多线性方程组是间接求解的，可以通过一些迭代方法来实现，例如Richardson法、Gauss-Seidel法、以及后续的过松弛法，或者Krylov子空间方法，如共轭梯度法、最小残差法或双共轭梯度法。我们参考了Stoer和Burlisch（2002年）、Strang（2003年）以及Liesen和Mehrmann（2015年）等书籍来进一步了解。

> Richardson method（理查德森外推法），the Jacobi method（雅可比法），the Gauß-Seidel method（高斯一赛德尔迭代法） the successive over-relaxation method（连续的逐次超松弛法 SOR）, Krylov subspace methods（维子空间方法，比如共轭梯度，广义极小残差），biconjugate gradients（双共轭梯度）

假设 $x_*$ 是方程 $Ax = b$ 的一个解，这些迭代法的关键思想是设置如下**迭代公式**：
\$$
x^{(k+1)} = Cx^{(k)} + d \tag{2.60}
\$$

其中，$C$ 和 $d$ 适当选择，能够在每次迭代中减少残差 $||x^{(k+1)} - x_*||$，并且收敛于 $x_*$。我们将在第3.1节中引入范数 $|| \cdot ||$，它允许我们计算向量间的相似性。

> 简单总结：
>
> - 对于小规模线性方程组，高斯消元法是一种直观且有效的方法。
> - 对于大规模问题，迭代法（如Jacobi法、Gauss-Seidel法、Krylov子空间方法等）更为实用，因为它们通过逐步逼近解来减少计算量。
> - 伪逆法提供了一种求解最小二乘问题的方式，但由于计算复杂度和数值精度问题，通常不推荐直接使用。
> - 迭代法的核心是通过构造迭代公式逐步减小残差，最终收敛到真实解。