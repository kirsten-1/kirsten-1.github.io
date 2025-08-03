---
layout: post
title: "scikit-learn 模型直接处理的非数值型数据"
date: 2025-08-03
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- 机器学习常见错误
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>



案例说明：利用KNN做分类任务“预测年收入是否大于50K美元”，读取adult.txt文件，最后一列是年收入，并使用KNN算法训练模型，然后使用模型预测一个人的年收入是否大于50

```python
df = pd.read_csv("../data/adults.txt", delimiter=",")
display(df)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

df内容如下：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250803135114315.png" alt="image-20250803135114315" style="zoom:50%;" />

在进行训练时，出现报错：

`ValueError: could not convert string to float: 'State-gov'` 是一个非常典型的机器学习数据预处理错误，它表明您的数据集中包含了无法被 `scikit-learn` 模型直接处理的**非数值型数据**（字符串）。

在机器学习中，大多数算法（包括KNN）都要求输入数据是数值型的。您的 `adults.txt` 文件中的数据，`State-gov` 明显是一个字符串，而 `KNeighborsClassifier` 模型无法在字符串上计算距离，因此在 `fit` 阶段报错。

----

# 数据编码

要解决这个问题，需要对数据进行预处理，将这些非数值型数据转换为数值型数据。这通常被称为**数据编码**。对于分类特征，有几种常见的编码方法：

## 1.独热编码 (One-Hot Encoding)

独热编码是一种将分类变量转换为一系列二进制特征的方法。对于每个分类特征，它会创建一个新的列，其值在对应类别为 `1`，其他为 `0`。

例如，对于 `workclass` 这一列，如果它有 `'State-gov'`, `'Private'`, `'Self-emp-not-inc'` 三个类别，独热编码会创建三列：`workclass_State-gov`、`workclass_Private` 和 `workclass_Self-emp-not-inc`。某一行数据是某个值，就将其对应位置置为1，其余位置置为0.

可以使用 `pandas` 的 `get_dummies` 函数或 `scikit-learn` 的 `OneHotEncoder`。

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 1. 读取数据
# 使用 header=None 让 Pandas 自动分配整数列名
# 确保 skipinitialspace=True 处理数据中的前导空格
df = pd.read_csv("../data/adults.txt", delimiter=",", header=None, skipinitialspace=True)

# 2. 找出所有非数值列
# 最后一列是标签，也属于非数值列，但我们不希望它被独热编码作为特征
# 我们可以先将标签列分离，然后再对剩余的特征进行独热编码
X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

# 找出特征中的所有非数值列
categorical_cols_X = X_df.select_dtypes(include=['object']).columns

# 3. 对特征进行独热编码
# pd.get_dummies 会自动创建新的列
X_processed = pd.get_dummies(X_df, columns=categorical_cols_X, dtype=int)

# 4. 对标签进行独热编码
# KNN分类器可以处理多维标签，但通常我们期望的是一维标签。
# 对于二分类问题，独热编码会创建两列，这会导致y变成一个二维数组，
# 这对于 KNeighborsClassifier 的 fit 方法来说是不合适的，会报错。
# 更好的方法是使用 LabelEncoder。如果坚持只用独热编码，
# 并且标签是二元的（例如 <=50K, >50K），我们可以选择其中一列作为标签。
# 例如，选择代表 '>50K' 的列作为我们的二元标签。
y_processed = pd.get_dummies(y_df, dtype=int)

# 5. 划分训练集和测试集
# 独热编码后的标签 y_processed 是一个多列的 DataFrame
# KNeighborsClassifier.fit() 方法需要一个一维数组作为标签
# 所以我们需要选择其中一列作为我们的二元标签
# 假设标签列名为 '<=50K' 和 '>50K'
# 我们可以选择其中一个作为我们的标签，例如 '>50K'
y_encoded = y_processed['>50K'] if '>50K' in y_processed.columns else y_processed[y_processed.columns[0]]


# 使用 train_test_split，设置 test_size=0.2，表示20%的数据作为测试集
# random_state=42 可以确保每次划分的结果都是一样的
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# 6. 训练模型
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("模型训练成功！")
print(f"训练集大小: {X_train.shape[0]} 条数据")
print(f"测试集大小: {X_test.shape[0]} 条数据")
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250803141311378.png" alt="image-20250803141311378" style="zoom:50%;" />



测试：

```python
# 测试与评估：
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率: {accuracy:.2f}")
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250803141848091.png" alt="image-20250803141848091" style="zoom:50%;" />

准备率有点低，可以尝试找到最佳的k值.此处省略。





## 2.标签编码 (Label Encoding)

标签编码是将每个类别映射为一个整数。例如，`'State-gov'` 变为 `0`，`'Private'` 变为 `1`，`'Self-emp-not-inc'` 变为 `2`。

这种方法在处理**无序分类变量**时通常不是一个好的选择，因为它会引入一个模型可能误解的**人为顺序**（例如，`2` 比 `0` 大，模型可能会认为 `'Self-emp-not-inc'` 比 `'State-gov'` "更好"或"更重要"）。但在处理**有序分类变量**时（例如，`'Low'`, `'Medium'`, `'High'`），这是一种合适的编码方式。

**注意**：对于 `adults.txt` 数据集，由于大多数分类特征（如 `workclass`, `marital-status`）是无序的，因此**独热编码**是更好的选择。

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 读取数据
# 使用 header=None 让 Pandas 自动分配整数列名
# 确保 skipinitialspace=True 处理数据中的前导空格
df = pd.read_csv("../data/adults.txt", delimiter=",", header=None, skipinitialspace=True)

# 2. 找出所有非数值列，并进行标签编码
# 标签编码适用于有序数据，但对于无序数据（如workclass），
# 可能会引入不恰当的顺序，但代码仍可运行。
# 找出所有 'object' 类型的列
categorical_cols = df.select_dtypes(include=['object']).columns

# 对所有分类列（包括标签列）进行标签编码
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 3. 划分特征和标签
# 最后一列是标签，现在已经被编码为整数
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. 划分训练集和测试集
# 使用 train_test_split，设置 test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练模型
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("模型训练成功！")
print(f"训练集大小: {X_train.shape[0]} 条数据")
print(f"测试集大小: {X_test.shape[0]} 条数据")


```

寻找最佳的k值：

```python
# 存储不同K值对应的准确率
k_values = range(1, 21)
accuracies = []

for k in k_values:
    # 实例化KNN分类器，n_neighbors为当前的k值
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # 使用训练数据拟合模型
    knn.fit(X_train, y_train)
    
    # 使用测试数据进行预测
    y_pred = knn.predict(X_test)
    
    # 计算并存储准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k={k}, 准确率是:{accuracy}")
    accuracies.append(accuracy)

# 绘制K值与准确率的关系图
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('K值与准确率')
plt.xlabel('K值')
plt.ylabel('准确率')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# 找到准确率最高的K值
best_k_index = np.argmax(accuracies)
best_k = k_values[best_k_index]
best_accuracy = accuracies[best_k_index]

print(f"\n最佳K值: {best_k}, 对应的准确率: {best_accuracy:.2f}")
```



<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250803145524223.png" alt="image-20250803145524223" style="zoom:50%;" />

只使用标签编码（Label Encoding）比独热编码（One-Hot Encoding）训练快很多的主要原因在于**数据维度**

独热编码会为每一个分类特征中的每一个唯一类别创建一个新的列。例如，`adults.txt` 数据集中的 `native-country` 特征有超过40个不同的国家。如果你对这个特征进行独热编码，它会从1列变成40多列。如果数据集有多个这样的高基数（unique值很多）分类特征，独热编码会使特征数量急剧增加。

标签编码将分类特征中的每一个类别映射为一个整数，例如：`'State-gov'` -> `0`, `'Private'` -> `1`。它不会增加新的列。一个包含分类特征的列，经过标签编码后仍然只有一列。

下面代码用了2种编码：

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 读取数据
# 使用 header=None 让 Pandas 自动分配整数列名
# 确保 skipinitialspace=True 处理数据中的前导空格
df = pd.read_csv("../data/adults.txt", delimiter=",", header=None, skipinitialspace=True)

# 2. 找出所有非数值列
# 注意：这里我们使用 df.select_dtypes() 来自动识别所有'object'类型的列
# 它们通常是字符串列
categorical_cols = df.select_dtypes(include=['object']).columns

# 3. 对所有非数值特征进行独热编码
# pd.get_dummies 会自动创建新的列
df_processed = pd.get_dummies(df, columns=categorical_cols, dtype=int)

# 4. 划分特征和标签
# 在原始数据中，最后一列是标签。在独热编码后，标签也可能被编码。
# 让我们假设原始的最后一列就是标签，并且它被独热编码了。
# 我们可以找到所有由原始最后一列独热编码而来的列。
# 为了简化，我们直接取最后一列作为标签，但请注意，如果原始标签有多个类别，这可能会有问题。
# 更好的方法是使用 LabelEncoder。

# 假设最后一列是原始标签
y_original = df.iloc[:, -1]
le = LabelEncoder()
y_encoded = le.fit_transform(y_original)

# 移除原始标签列，得到特征矩阵 X
X = df_processed.drop(y_original.name, axis=1, errors='ignore')

# 5. 划分训练集和测试集
# 使用 train_test_split，设置 test_size=0.2，表示20%的数据作为测试集
# random_state=42 可以确保每次划分的结果都是一样的
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. 训练模型
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("模型训练成功！")
print(f"训练集大小: {X_train.shape[0]} 条数据")
print(f"测试集大小: {X_test.shape[0]} 条数据")

# 7. (可选) 在测试集上进行预测和评估
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率: {accuracy:.2f}")
```

用于训练模型的特征矩阵 `X` 是经过独热编码的，而标签向量 `y_encoded` 是经过标签编码的。这是处理分类问题时一种常见的混合策略，特别是在处理多分类或二分类标签时。

训练与测试结果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250803150500871.png" alt="image-20250803150500871" style="zoom:50%;" />





















