---
layout: post
title: "【java集合】02.Collection接口"
date: 2025-09-04
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- java集合
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>




Collection 接口是 Java 集合框架的根接口之一，它定义了所有单值集合（如 `List` 和 `Set`）的通用行为。了解它的常用方法是学习集合的第一步。

---

# 1.Collection常用方法

以下是 `Collection` 接口中一些最常用且重要的方法，可以帮助你进行基本的集合操作：

## 1. 添加元素

- `boolean add(E e)` 用于向集合中添加一个元素。如果集合因调用此方法而发生改变，则返回 `true`。需要注意的是，`Set` 集合会忽略重复的添加请求，并返回 `false`。
- `boolean addAll(Collection<? extends E> c)` 将指定集合中的所有元素都添加到当前集合中。如果集合因此调用而发生改变，则返回 `true`。

【特别注意】集合有一个特点：只能存放引用数据类型的数据，不能是基本数据类型

所以在集合中存储的元素往往会自动装箱（Autoboxing）。

## 2. 删除元素

- `boolean remove(Object o)` 从集合中移除指定元素的单个实例（如果存在）。如果成功移除，则返回 `true`。
- `boolean removeAll(Collection<?> c)` 从集合中移除与指定集合中所有元素都相同的元素。简单来说，就是取两个集合的差集。如果集合因此调用而发生改变，则返回 `true`。
- `boolean retainAll(Collection<?> c)` 仅保留此集合中那些也包含在指定集合中的元素。这是一种取 **交集** 的操作。如果集合因此调用而发生改变，则返回 `true`。
- `void clear()` 移除集合中的所有元素，使其变为空集合。

## 3. 查询和判断

- `int size()` 返回集合中元素的数量。
- `boolean isEmpty()` 如果集合不包含任何元素，则返回 `true`。
- `boolean contains(Object o)` 如果集合包含指定的元素，则返回 `true`。
- `boolean containsAll(Collection<?> c)` 如果集合包含指定集合中的所有元素，则返回 `true`。

## 4. 转换

- `Object[] toArray()` 返回一个包含集合中所有元素的数组。返回的数组类型是 `Object[]`。
- `E[] toArray(T[] a)` 返回一个包含集合中所有元素的数组，并指定返回数组的类型。这是一个泛型方法，提供了更好的类型安全性。

## 5. 遍历

- `Iterator<E> iterator()` 返回一个在此集合的元素上进行迭代的迭代器（`Iterator`）。这是遍历集合最基础、最安全的方式。

### 迭代器模式

迭代器（Iterator）模式，也叫做**游标（Cursor）模式**。

在Java 容器中，为了提高容器遍历的方便性，利用迭代器把遍历逻辑从不同类型的集合类中抽取出来，从而避免向外部暴露集合容器的内部结构。

Java 提供了 `Iterator` 接口作为迭代器的基础接口。该接口定义了一组用于访问集合元素的方法，包括 `hasNext`、`next` 和 `remove` 等。

```java
// 迭代器
Iterator iterator = collection.iterator();
while (iterator.hasNext()) {
    System.out.print(iterator.next() + " ");
}
System.out.println();


// 遍历：增强for循环
for (Object i : collection) {
    System.out.print(i + " ");
}
System.out.println();
```

![image-20250904152750890](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250904152750890.png)

# 2.使用上的注意点

## 2.1集合判空的推荐方法-isEmpty

**判断所有集合内部的元素是否为空，使用 `isEmpty()` 方法，而不是 `size()==0` 的方式。**

这是因为 `isEmpty()` 方法的可读性更好，并且效率更高。

`isEmpty()`时间复杂度为 `O(1)`。集合的底层实现通常会维护一个 `size` 变量来记录元素的数量。每当添加或移除元素时，这个变量就会被更新。因此，`size()` 和 `isEmpty()` 方法都只需要直接返回这个变量的值，**时间复杂度都是 O(1)**。

`size()` 方法需要遍历整个链表，时间复杂度为`O(n)`。（尽管在现代 Java 中，大多数集合的 `size()` 方法时间复杂度也为 `O(1)`，但依然强烈建议使用 `isEmpty()`）`isEmpty()` 方法的设计初衷就是为了这个特定的目的，它的语义比 `size() == 0` 更加精确和直接。



# 3.为什么集合只能存放引用数据类型？

这个限制源于 Java 集合框架的设计。集合框架（如 `ArrayList`、`HashSet`、`HashMap` 等）是为 **面向对象编程** 设计的。所有集合类都继承或实现了 `Collection`、`List`、`Set`、`Map` 等接口，这些接口的方法都接受或返回 `Object` 类型，或者在泛型中使用类型参数（`E`、`K`、`V` 等）。

- **内存存储方式**：Java 集合存储的是对象的 **引用**（内存地址），而不是对象本身。这意味着集合内部维护的是一个指向对象的指针列表。基本数据类型（如 `int`、`double`、`boolean`）不是对象，它们直接存储在栈上或特定的内存区域，没有引用的概念。因此，集合不能直接存储它们。



