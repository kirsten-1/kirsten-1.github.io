---
layout: post
title: "jvm-class文件格式"
subtitle: "在学习jvm的各种内容之前应该简单了解下class文件的格式。"
date: 2025-06-05
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
- jvm
---

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">
</script>


官网关于ClassFileFormat（JDK8）：https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-4.html

-----

# 1 一个最简单的案例开始

准备一个最简单的java源代码：

```java
public class SimpleExample {
}
```

利用idea的build工具进行编译：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603133759170.png" alt="image-20250603133759170" style="zoom:50%;" />

在`out/production`目录中找到`SimpleExample.class`字节码文件。

利用javap进行反编译，并且将内容输出为一个txt文件：

```java
javap -p -v SimpleExample.class >example.txt 
```

`example.txt`内容如下：

```java
Classfile /Users/apple/Documents/study/JVM_Exer/out/production/E_ClassFileFormat/SimpleExample.class
  Last modified 2025年6月3日; size 264 bytes
  SHA-256 checksum 062d6c4064cb0c855239a3a067596316804edc006b2232a562e6bdb325ecb5b8
  Compiled from "SimpleExample.java"
public class SimpleExample
  minor version: 0
  major version: 52
  flags: (0x0021) ACC_PUBLIC, ACC_SUPER
  this_class: #2                          // SimpleExample
  super_class: #3                         // java/lang/Object
  interfaces: 0, fields: 0, methods: 1, attributes: 1
Constant pool:
   #1 = Methodref          #3.#13         // java/lang/Object."<init>":()V
   #2 = Class              #14            // SimpleExample
   #3 = Class              #15            // java/lang/Object
   #4 = Utf8               <init>
   #5 = Utf8               ()V
   #6 = Utf8               Code
   #7 = Utf8               LineNumberTable
   #8 = Utf8               LocalVariableTable
   #9 = Utf8               this
  #10 = Utf8               LSimpleExample;
  #11 = Utf8               SourceFile
  #12 = Utf8               SimpleExample.java
  #13 = NameAndType        #4:#5          // "<init>":()V
  #14 = Utf8               SimpleExample
  #15 = Utf8               java/lang/Object
{
  public SimpleExample();
    descriptor: ()V
    flags: (0x0001) ACC_PUBLIC
    Code:
      stack=1, locals=1, args_size=1
         0: aload_0
         1: invokespecial #1                  // Method java/lang/Object."<init>":()V
         4: return
      LineNumberTable:
        line 1: 0
      LocalVariableTable:
        Start  Length  Slot  Name   Signature
            0       5     0  this   LSimpleExample;
}
SourceFile: "SimpleExample.java"

```

## 魔术与版本号

首先`SHA-256 checksum 062d6c...`是文件的`SHA-256`校验和，用于验证文件完整性。

`Compiled from "SimpleExample.java"`: 表明这个`.class`文件是由`SimpleExample.java`源代码编译而来。

`public class SimpleExample`: 对应源代码中的`public class SimpleExample {}`，表示这是一个公共类。

这些内容都不用重点关注。

----

**`minor version: 0`** 和 **`major version: 52`**:

- `major version: 52` 对应 Java SE 8 (JDK 1.8)。这意味着该类文件是由JDK 1.8或更高版本编译的，并且可以在JDK 1.8或更高版本的JVM上运行。
- `minor version` 通常在主版本号不变的情况下进行微小的更新，这里是0。

根据官网，class文件结构其实是遵循下面这样的结构的：

```java
ClassFile {
    u4             magic;
    u2             minor_version;
    u2             major_version;
    u2             constant_pool_count;
    cp_info        constant_pool[constant_pool_count-1];
    u2             access_flags;
    u2             this_class;
    u2             super_class;
    u2             interfaces_count;
    u2             interfaces[interfaces_count];
    u2             fields_count;
    field_info     fields[fields_count];
    u2             methods_count;
    method_info    methods[methods_count];
    u2             attributes_count;
    attribute_info attributes[attributes_count];
}
```

`u4`代表unsigned 4字节，同理`u2`就代表unsigned 2字节。

一开始4个字节是`magic`，一般翻译成魔术，是固定的4个字节的值——`cafe babe`:

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603134819416.png" alt="image-20250603134819416" style="zoom:40%;" />

在上面的文件中`minor_version`就是`0000`,`major version`就是`0034`。而34（十六进制，3*16+4=52）变成十进制就是52，就代表是`JDK 1.8`。

-----

## 常量池

`constant_pool_count`:常量池计数器,上图中是`0010`，即十进制的16，其实真正常量池中应该有15个。官网详细说明了：

> The value of the `constant_pool_count` item is equal to the number of entries in the `constant_pool` table plus one. A `constant_pool` index is considered valid if it is greater than zero and less than `constant_pool_count`, with the exception for constants of type `long` and `double` noted in [§4.4.5](https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-4.html#jvms-4.4.5).
>
> `Constant_pool_count `项的值等于 `constant_pool `表中的条目数加 1。如果 `constant_pool` 索引大于 0 且小于 `constant_pool_count`, 则视为有效，但 4.4.5 节中提到的 long 类型和 double 类型的常量除外。

反编译的结果中也验证了这一点：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603135358607.png" alt="image-20250603135358607" style="zoom:50%;" />

那么，**常量池中最多可以有几个？**因为是2字节，又要减1，所以是$2^{16}-1$。

-----

然后就是`cp_info`类型的`constant_pool[constant_pool_count-1];`。`Constant_pool` 是一个结构表, 表示 ClassFile 结构及其子结构中引用的各种字符串常量、类和接口名称、字段名称以及其他常量。每个 `constant_pool` 表项的格式由其第一个 “tag” 字节指示。`Constant_pool` 表的索引从 1 到 `constant_pool_count-1`(上图中就是从`#1`到`#15`)。

> tag是什么？是常量池表项的第一个字节，指示表项的类型（如 CONSTANT_Utf8、CONSTANT_Class、CONSTANT_Methodref 等）。
>
> 下面是所有的类型罗列：
>
> | tag  | 常量类型 (cp_info)               | 描述                                                         |
> | ---- | -------------------------------- | ------------------------------------------------------------ |
> | 1    | CONSTANT_Utf8_info               | length: 2字节，表示 UTF-8 编码字符串的字节数 bytes: 长度为 length 的字节序列 |
> | 3    | CONSTANT_Integer_info            | bytes: 4字节，Big-Endian（高位在前）存储的整数值             |
> | 4    | CONSTANT_Float_info              | bytes: 4字节，Big-Endian（高位在前）存储的 IEEE 754 浮点数   |
> | 5    | CONSTANT_Long_info               | bytes: 8字节，Big-Endian（高位在前）存储的长整数值           |
> | 6    | CONSTANT_Double_info             | bytes: 8字节，Big-Endian（高位在前）存储的 IEEE 754 双精度浮点数 |
> | 7    | CONSTANT_Class_info              | index: 2字节，指向常量池中某个 CONSTANT_Utf8_info 表项，表示类或接口名 |
> | 8    | CONSTANT_String_info             | index: 2字节，指向常量池中某个 CONSTANT_Utf8_info 表项，表示字符串 |
> | 9    | CONSTANT_Fieldref_info           | index: 2字节，指向声明字段的类或接口的 CONSTANT_Class_info 表项 index: 2字节，指向 CONSTANT_NameAndType_info 表项 |
> | 10   | CONSTANT_Methodref_info          | index: 2字节，指向声明方法的类的 CONSTANT_Class_info 表项 index: 2字节，指向 CONSTANT_NameAndType_info 表项 |
> | 11   | CONSTANT_InterfaceMethodref_info | index: 2字节，指向声明接口方法的接口的 CONSTANT_Class_info 表项 index: 2字节，指向 CONSTANT_NameAndType_info 表项 |
> | 12   | CONSTANT_NameAndType_info        | index: 2字节，指向字段或方法的名称的 CONSTANT_Utf8_info 表项 index: 2字节，指向字段或方法的描述符的 CONSTANT_Utf8_info 表项 |
> | 15   | CONSTANT_MethodHandle_info       | reference_kind: 1字节，1-9 之间的值，表示引用的类型（如方法调用、字段访问等） reference_index: 2字节，指向常量池中对应的表项 |
> | 16   | CONSTANT_MethodType_info         | descriptor_index: 2字节，指向 CONSTANT_Utf8_info 表项，表示方法的描述符 |
> | 18   | CONSTANT_InvokeDynamic_info      | bootstrap_method_attr_index: 2字节，指向 Class 文件中 BootstrapMethods 属性的表项 name_and_type_index: 2字节，指向 CONSTANT_NameAndType_info 表项，表示方法名和描述符 |
>
> 表格中的每一行对应一个 tag 值及其对应的 cp_info 结构。

例如对于刚才反编译的结果：

```java
   #1 = Methodref          #3.#13         // java/lang/Object."<init>":()V
   #2 = Class              #14            // SimpleExample
   #3 = Class              #15            // java/lang/Object
   #4 = Utf8               <init>
   #5 = Utf8               ()V
   #6 = Utf8               Code
   #7 = Utf8               LineNumberTable
   #8 = Utf8               LocalVariableTable
   #9 = Utf8               this
  #10 = Utf8               LSimpleExample;
  #11 = Utf8               SourceFile
  #12 = Utf8               SimpleExample.java
  #13 = NameAndType        #4:#5          // "<init>":()V
  #14 = Utf8               SimpleExample
  #15 = Utf8               java/lang/Object
```

第一个是`Methodref`类型，对应tag=10,表示：指向声明方法的类的 CONSTANT_Class_info 表项（2字节），指向 CONSTANT_NameAndType_info 表项（2字节）。这里指向了`#3`和`#13`，对应的是

```java
#3 = Class              #15            // java/lang/Object
#13 = NameAndType        #4:#5          // "<init>":()V
```

`#3`是`Object`类，而`#13`对应的是`"<init>":()V`，其中`()`代表的是空的参数列表，V代表返回值类型是`void`，其实这个`"<init>":()V`就是空的构造方法，表示当前类（`SimpleExample`）的构造器将调用其父类`java.lang.Object`的无参构造器。（我们都知道当一个类被实例化时，java语言规范要求在调用子类构造器之前，必须先调用其父类的构造器）。

-----

接着往下就是

**`#2 = Class #14 // SimpleExample`**: 这是一个类引用，指向常量池中的`#14`，它是一个UTF-8字符串`SimpleExample`，表示当前类的名称。

**`#3 = Class #15 // java/lang/Object`**: 这是一个类引用，指向常量池中的`#15`，它是一个UTF-8字符串`java/lang/Object`，表示父类的名称。

**`#4 = Utf8 <init>`**: UTF-8字符串，表示特殊方法名`<init>`，这是Java中实例构造器的名称。

**`#5 = Utf8 ()V`**: UTF-8字符串，表示方法的描述符。`()`表示无参数，`V`表示返回类型为`void`。所以`()V`表示一个无参数且无返回值的构造器或方法。

-----

然后就到了code 属性，这是决定代码逻辑最重要的东西：code属性是方法信息（method_info）结构中的一个可选属性，用于存储方法的字节码以及相关的执行信息。它主要出现在非抽象、非本地（non-native）方法的定义中，例如 SimpleExample.class 中 public SimpleExample() 方法的 Code 属性部分。下面是`#6`是UTF-8字符串，表示一个方法属性的名称，即`Code`属性，它包含了方法的字节码指令。

```java
#6 = Utf8               Code
```

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603152051318.png" alt="image-20250603152051318" style="zoom:50%;" />

在后面会有详细的code属性。后面再解释，先把所有的常量池大致看完：

**`#7 = Utf8 LineNumberTable`**: UTF-8字符串，表示一个方法属性的名称，即`LineNumberTable`属性，用于将字节码指令的偏移量映射到源代码的行号。

**`#8 = Utf8 LocalVariableTable`**: UTF-8字符串，表示一个方法属性的名称，即`LocalVariableTable`属性，用于描述方法中的局部变量。

**`#9 = Utf8 this`**: UTF-8字符串，表示局部变量的名称`this`。

**`#10 = Utf8 LSimpleExample;`**: UTF-8字符串，表示局部变量的类型描述符。`L`表示这是一个对象类型，`SimpleExample;`是其全限定名，分号表示结束。所以`LSimpleExample;`表示类型为`SimpleExample`的对象。

**`#11 = Utf8 SourceFile`**: UTF-8字符串，表示一个类文件属性的名称，即`SourceFile`属性。

**`#12 = Utf8 SimpleExample.java`**: UTF-8字符串，表示源文件的名称。

**`#13 = NameAndType #4:#5 // "<init>":()V`**: 这是一个名称和类型描述符的组合。

- 它引用了常量池中的`#4`（名称`<init>`）和`#5`（描述符`()V`）。
- 组合起来表示一个无参无返回值的构造器。

**`#14 = Utf8 SimpleExample`**: UTF-8字符串，表示类名`SimpleExample`。

**`#15 = Utf8 java/lang/Object`**: UTF-8字符串，表示类名`java/lang/Object`。

-----

## Method

常量池之后会有这样一块内容：

```java
{
  public SimpleExample();
    descriptor: ()V
    flags: (0x0001) ACC_PUBLIC
    Code:
      stack=1, locals=1, args_size=1
         0: aload_0
         1: invokespecial #1                  // Method java/lang/Object."<init>":()V
         4: return
      LineNumberTable:
        line 1: 0
      LocalVariableTable:
        Start  Length  Slot  Name   Signature
            0       5     0  this   LSimpleExample;
}
```

显示了类中定义的所有方法。由于`SimpleExample`类没有显式定义任何方法，编译器为其生成了一个默认的无参构造器`public SimpleExample();`。

然后这个构造方法分成了3部分描述：

- **`descriptor: ()V`**: 方法的描述符。`()`表示该构造器不接受任何参数（参数列表是空），`V`表示返回类型为`void`（构造器没有显式返回值）。
- **`flags: (0x0001) ACC_PUBLIC`**: 表示该构造器是公共的，可以被其他类访问。

这里需要补充下flags.

### flags

在这个类中也有flags：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603153300074.png" alt="image-20250603153300074" style="zoom:50%;" />

构造方法中也有flags:

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603153327386.png" alt="image-20250603153327386" style="zoom:50%;" />

flags 字段用于描述类、方法或字段的访问权限和属性。flags 是一个位掩码（bitmask），通过按位或（|）组合多个标志位来表示不同的属性。每个标志位对应一个特定的二进制位，值的计算基于这些位的组合。

首先说明：

- `ACC_PUBLIC`：表示类是公共的（public），可以被其他类访问。
- `ACC_SUPER`：一个历史遗留标志，用于控制 invokespecial 指令的行为（在现代 JVM 中通常被忽略，但编译器会默认设置）。
    - 可以简单理解为：`ACC_SUPER`一般的类都会有，jvm默认都是有这个值的。（JDK1.0.2之后编译出来的类的这个标志都必须为真）

其实不止这2个取值，根据 Java 虚拟机规范（JVM Specification），类的访问标志（`access_flags`）是 16 位（2 字节），每个标志位对应一个特定的值（以十六进制表示）。

下面是所有的取值可能：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603153750021.png" alt="image-20250603153750021" style="zoom:50%;" />

可以看到`ACC_SUPER`标志值是`0x0020`，而`ACC_PUBLIC`标志值是`0x0001`，如何计算得到`0x0021`的呢？

按位或即可：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250603154834303.png" alt="image-20250603154834303" style="zoom:50%;" />

----

### code

code包含了方法的字节码指令以及其他与代码执行相关的信息

```java
    Code:
      stack=1, locals=1, args_size=1
         0: aload_0
         1: invokespecial #1                  // Method java/lang/Object."<init>":()V
         4: return
      LineNumberTable:
        line 1: 0
      LocalVariableTable:
        Start  Length  Slot  Name   Signature
            0       5     0  this   LSimpleExample;
```

- `stack=1`: 操作数栈的最大深度为1。在执行过程中，最多需要1个栈帧来存储数据。
- `locals=1`: 局部变量表的大小为1。局部变量表用于存储方法的参数和局部变量。
- `args_size=1`: 方法的参数数量为1。对于实例方法（包括构造器），第一个局部变量槽（Slot 0）总是用来存储`this`引用。所以这里的`args_size=1`表示除了`this`之外没有其他显式参数。

后面是3条指令：

```java
         0: aload_0
         1: invokespecial #1                  // Method java/lang/Object."<init>":()V
         4: return
```

关于常见的指令总结在最后附录中。

解释这里3条指令：

- `0: aload_0`：冒号前面的`0` 是字节码指令的偏移量（offset）。`aload_0` 指令将局部变量表中索引为0的引用类型变量加载到操作数栈顶。在实例方法和构造器中，局部变量表的Slot 0 总是用于存储`this`引用。所以这条指令的作用是将当前对象（`this`）的引用压入操作数栈。
- `1: invokespecial #1  // Method java/lang/Object."<init>":()V`:`1` 是字节码指令的偏移量。
    - `invokespecial` 指令用于调用实例初始化方法（构造器）、私有方法和父类方法。
    - `#1` 指向常量池中索引为1的项，即`java/lang/Object."<init>":()V`，表示调用`java.lang.Object`类的无参构造器。
    - 在Java中，每个构造器的第一行（如果没有显式调用`this()`或`super()`）都会隐式调用父类的无参构造器。

- **`4: return`**:

    - `4` 是字节码指令的偏移量。

    - `return` 指令用于从方法中返回。对于`void`方法或构造器，它只是简单地结束方法的执行。

------

画图说明这3行指令的过程：

首先要注意：在任何Java实例方法或构造器开始执行时，JVM 会为该方法创建一个 **栈帧（Stack Frame）**。栈帧中包含：

1. **局部变量表 (Local Variable Table)**：用于存储方法的参数和局部变量。对于实例方法，第一个槽位（Slot 0）总是用来存储 `this` 引用。
2. **操作数栈 (Operand Stack)**：一个基于栈的临时存储区域，用于存储计算过程中的中间结果以及方法调用的参数和返回值。

当 JVM 调用 `SimpleExample` 构造器时，会创建一个新的栈帧。局部变量表中的 Slot 0 会被填充为当前 `SimpleExample` 实例的引用（即 `this`），操作数栈为空。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250604155134474.png" alt="image-20250604155134474" style="zoom:50%;" />

`aload_0` 将局部变量表索引为 0 的引用类型变量（即 `this` 引用）加载到操作数栈顶。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250604155442153.png" alt="image-20250604155442153" style="zoom:50%;" />

`invokespecial` 用于调用父类的构造器。它会从操作数栈顶弹出一个对象引用（这里是 `this` 引用），作为隐式参数传递给被调用的方法（`java.lang.Object` 的 `<init>` 方法），然后执行该方法。

在这个例子中，`#1` 指向 `java/lang/Object."<init>":()V`，表示调用 `Object` 类的无参构造器。`invokespecial` 会消耗栈顶的 `this` 引用。

执行完 `invokespecial` 后，PC 会更新为下一条指令的偏移量 `4`。

**栈帧变化：** 操作数栈顶的 `this` 引用被消耗（弹出）。在 `Object.<init>()V` 执行期间，JVM 会创建并压入一个新的栈帧到JVM栈上（这里未显示，因为它属于另一个方法调用），当 `Object.<init>()V` 执行完毕后，该栈帧会被弹出，控制权返回给 `SimpleExample.<init>()` 的栈帧。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250604155642940.png" alt="image-20250604155642940" style="zoom:50%;" />

`4: return` 表示从方法中返回。对于 `void` 方法或构造器，它不需要返回任何值。

- **程序计数器 (PC) 变化：** `return` 指令执行后，当前方法结束，PC 的值通常会变得不确定或指向调用者的下一条指令（取决于JVM实现，但对于当前方法而言，其PC不再有效）。
- **栈帧变化：** 当前 `SimpleExample.<init>()` 的栈帧被销毁，并从JVM栈中弹出。

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250604155918812.png" alt="image-20250604155918812" style="zoom:50%;" />

## 方法描述符补充

上面已经看到对于参数列表是空，返回值是void的方法（包括构造方法），方法描述符是`()V`

如果是其他情况呢？下面是字段描述符：

| FieldType 中的字符 | 类型      | 含义                          |
| ------------------ | --------- | ----------------------------- |
| B                  | byte      | 有符号的字节型数              |
| C                  | char      | Unicode 字符码点，UTF-16 编码 |
| D                  | double    | 双精度浮点数                  |
| F                  | float     | 单精度浮点数                  |
| I                  | int       | 整型数                        |
| J                  | long      | 长整型                        |
| L *className*;     | reference | *ClassName* 类的实例          |
| S                  | short     | 有符号短整型                  |
| Z                  | boolean   | 布尔值 true/false             |
| [                  | Reference | 一个一维数组                  |

如 String 类的实例，其描述符为 Ljava/lang/String。二维数组 int [][] 类型的实例变量，其描述符为 [[I。

看一道习题：

如果一个方法需要接收一个`int`类型的参数并返回一个`String`对象，它的方法描述符会是什么样子？

A.(I)LString;

B.(I)Ljava/lang/String;

C.(I)S

D.(int)String

> 答案：B
>
> 首先int对应FieldType 中的字符中的`I`，其次String对应的是`Ljava/lang/String`



# 附录

## 1.常见jvm指令

关于jvm的指令集（一共也就二百多条指令）官网：https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-4.html#jvms-4.10.1.9

下面是常见的总结：

| 类型 | 助记符          | 含义                                                         |
| ---- | --------------- | ------------------------------------------------------------ |
| 常量 | const_null      | 将 null 推到栈顶                                             |
|      | iconst_1        | 将整数类型的 1 推到栈顶                                      |
|      | fconst_0        | 将 float 类型的 2 推到栈顶                                   |
|      | dcounst_0       | 将 double 类型的 0 推到栈顶                                  |
|      | ldc             | 将 int 、float 或 String 类型常量值从常量池推至栈顶          |
|      | …               |                                                              |
| 加载 | iload           | 将指定的 int 类型本地变量推送至栈顶                          |
|      | iload_2         | 第 2 个 int 类型本地变量推送至栈顶                           |
|      | aload_3         | 将第 3 个引用类型的本地变量推送至栈顶                        |
|      | cload           | 将 char 类型数组的指定元素推送至栈顶                         |
|      | …               |                                                              |
| 存储 | istore          | 将栈顶 int 类型数值存入指定的本地变量                        |
|      | astore          | 将栈顶引用类型的数值存入指定的本地变量                       |
|      | istore_3        | 将栈顶 int 类型数值存入第 3 个本地变量                       |
|      | …               |                                                              |
| 引用 | getstatic       | 获取指定类的静态字段，并将其压入栈顶                         |
|      | putstatic       | 为指定类的静态字段赋值                                       |
|      | getfield        | 获取指定类的实例字段，并将其值压入栈顶                       |
|      | putfield        | 为指定类的实例字段赋值                                       |
|      | invokevirtual   | 调用实例方法                                                 |
|      | invokespecial   | 调用父方法、实例初始化方法、私有方法                         |
|      | invokestatic    | 调用静态方法                                                 |
|      | invokeinterface | 调用接口方法                                                 |
|      | invokedynamic   | 调用动态连接方法                                             |
|      | new             | 创建一个对象，并将其引用压入栈顶                             |
|      | athrow          | 将栈顶异常抛出                                               |
|      | instanceof      | 检查对象是否为指定类的实例，如果是，将 1 压到栈顶，否则将 0 压到栈顶 |
|      | …               |                                                              |
| 栈   | pop             | 将栈顶数值弹出（非 long 和 double）                          |
|      | pop2            | 将栈顶 long 或 double 数值弹出                               |
|      | dup             | 复制栈顶数值并将复制值压入栈顶                               |
|      | …               |                                                              |
| 控制 | ireturn         | 从当前方法返回 int 值                                        |
|      | return          | 从当前方法返回 void 值                                       |
|      | …               |                                                              |
| 比较 | ifeq            | 当栈顶的 int 类型的数值等于 0 时跳转                         |
|      | ifne            | 当栈顶的 int 类型的数值不等于 0 时跳转                       |
|      | …               |                                                              |
| 拓展 |                 | 为 null 时跳转                                               |
|      | …               |                                                              |
| 数学 | …               |                                                              |
| 转换 |                 |                                                              |
| 比较 |                 |                                                              |

## 2.习题

1. 字节码指令 `aload_0` 在这个构造器中的作用是什么？

   A.将 `this` 引用从局部变量表加载到操作数栈。

   B.分配内存给一个新对象。

   C.调用地址为0的方法。

   D.将整数0加载到操作数栈。

2. `.class` 文件中的 `SourceFile` 属性有什么用途？

   A.指定该类文件依赖的其他源文件。

   B.在抛出异常时，用于在堆栈跟踪中显示源文件名和行号。

   C.包含整个Java源代码，以便JVM可以动态编译它。

   D.为编译器提供一个唯一的标识符。

3. 在JVM的运行时数据区中，哪个区域是所有线程共享的？

   A.Java虚拟机栈 (JVM Stack)

   B.方法区 (Method Area)

   C.本地方法栈 (Native Method Stack)

   D.程序计数器 (PC Register)

4. 阅读下面javap -p -v的反编译结果，在`javap`输出中，`this_class: #2` 这个条目指向常量池中的什么内容？

   ```java
   Classfile /Users/apple/Documents/study/JVM_Exer/out/production/E_ClassFileFormat/SimpleExample.class
     Last modified 2025年6月3日; size 264 bytes
     SHA-256 checksum 062d6c4064cb0c855239a3a067596316804edc006b2232a562e6bdb325ecb5b8
     Compiled from "SimpleExample.java"
   public class SimpleExample
     minor version: 0
     major version: 52
     flags: (0x0021) ACC_PUBLIC, ACC_SUPER
     this_class: #2                          // SimpleExample
     super_class: #3                         // java/lang/Object
     interfaces: 0, fields: 0, methods: 1, attributes: 1
   Constant pool:
      #1 = Methodref          #3.#13         // java/lang/Object."<init>":()V
      #2 = Class              #14            // SimpleExample
      #3 = Class              #15            // java/lang/Object
      #4 = Utf8               <init>
      #5 = Utf8               ()V
      #6 = Utf8               Code
      #7 = Utf8               LineNumberTable
      #8 = Utf8               LocalVariableTable
      #9 = Utf8               this
     #10 = Utf8               LSimpleExample;
     #11 = Utf8               SourceFile
     #12 = Utf8               SimpleExample.java
     #13 = NameAndType        #4:#5          // "<init>":()V
     #14 = Utf8               SimpleExample
     #15 = Utf8               java/lang/Object
   {
     public SimpleExample();
       descriptor: ()V
       flags: (0x0001) ACC_PUBLIC
       Code:
         stack=1, locals=1, args_size=1
            0: aload_0
            1: invokespecial #1                  // Method java/lang/Object."<init>":()V
            4: return
         LineNumberTable:
           line 1: 0
         LocalVariableTable:
           Start  Length  Slot  Name   Signature
               0       5     0  this   LSimpleExample;
   }
   SourceFile: "SimpleExample.java"
   
   ```

   A.一个指向当前类的`Class`类型常量。

   B.一个指向父类的`Class`类型常量。

   C.当前类的实例（`this`引用）。

   D.一个名为`this_class`的UTF-8字符串。

5. 在Java中，如果一个类显式地定义了一个带参数的构造器，编译器还会自动生成一个无参构造器吗？

   A.会，但会将其设为`private`。

   B.不会，一旦用户定义了任何构造器，编译器就不再提供默认的无参构造器。

   C.会，编译器总是会确保有一个无参构造器。

   D.这取决于构造器是否是`public`的。

6. 在`Code`属性的字节码中，`return`指令的作用是什么？

   A.结束当前方法的执行并销毁当前栈帧。

   B.清空操作数栈和局部变量表。

   C.将操作数栈顶的值返回给调用者。

   D.跳转到方法的起始位置，重新执行。

7. `.class`文件中的`LineNumberTable`属性的主要作用是什么？

   A.存储每个源代码行的文本内容。

   B.优化代码，移除不必要的代码行。

   C.定义类文件中总共有多少行代码。

   D.将字节码指令的偏移量映射到Java源代码的行号。

8. 程序计数器（PC Register）中存储的是什么？

   A.当前线程正在执行的字节码指令的地址（或偏移量）。

   B.当前操作数栈顶元素的值。

   C.下一个要执行的Java方法的内存地址。

   D.局部变量表中`this`引用的地址。

9. 在JVM中，“栈帧”（Stack Frame）是在什么时候被创建的？

   A.当一个对象被创建时（`new`）。

   B.当一个线程启动时。

   C.当一个类被加载时。

   D.当一个方法被调用时。

10. 如果一个方法需要接收一个`int`类型的参数并返回一个`String`对象，它的方法描述符会是什么样子？

    A.(I)LString;

    B.(I)Ljava/lang/String;

    C.(I)S

    D.(int)String

11. 字节码指令`iconst_1`的作用是什么？(可查附录1的表)

    A.创建一个包含一个元素的整型数组。

    B.将局部变量表中索引为1的整数加载到操作数栈。

    C.将常量1压入操作数栈。

    D.检查栈顶的整数是否等于1。

12. 在常量池中，`Utf8`类型的条目存储的是什么？

    A.一个32位的整数常量。

    B.一个指向其他常量池条目的索引。

    C.一个编译后的方法代码。

    D.一个UTF-8编码的字符串值。

13. 以下哪项是JVM中堆（Heap）区域的主要特点？

    A.它是所有线程共享的，用于存储对象实例和数组。

    B.它是线程私有的，用于存储局部变量。

    C.它存储了下一条要执行的字节码指令的地址。

    D.它主要用于存储运行时常量池。

14. 在`javap`输出中，`flags: (0x0001) ACC_PUBLIC` 在方法描述符中代表什么？

    A.该方法是公共的，可以从任何类访问。

    B.该方法是静态的。

    C.该方法可以被重写。

    D.该方法是抽象的。

15. 以下哪个字节码指令用于创建新的对象实例？

    A.`invokevirtual`

    B.`astore_0`

    C.`new`

    D.`putfield`

16. 方法描述符 `(Ljava/lang/String;I)V` 意味着什么？

    A.一个接受一个`String`参数和一个`int`参数，并且没有返回值的方法。

    B.一个接受一个`String`参数并返回一个`int`的方法。

    C.一个接受两个`String`参数的方法。

    D.一个返回`String`类型的方法，没有参数。

17. 在JVM中，为什么局部变量表中的Slot 0通常被`this`引用占据（对于实例方法而言）？

    A.因为`this`引用是唯一不需要初始化的局部变量。

    B.为了防止栈溢出错误。

    C.为了节省内存空间，因为`this`引用是最小的。

    D.为了在方法内部能够访问当前对象实例的成员。

18. 哪个JVM运行时数据区是线程私有的，用于存储每个方法调用的栈帧？

    A.Java虚拟机栈 (JVM Stack)

    B.运行时常量池 (Runtime Constant Pool)

    C.方法区 (Method Area)

    D.堆 (Heap)

19. 在常量池中，`NameAndType`类型的条目主要用于什么？

    A.指示一个方法是静态的还是实例的。

    B.存储字面量字符串。

    C.定义一个类的完整限定名。

    D.将字段或方法的名称和描述符组合起来。

20. 字节码指令`getfield`的作用是什么？

    A.调用一个方法。

    B.获取静态字段的值。

    C.获取数组中指定索引的元素。

    D.获取对象实例字段的值。

21. 在Java中，`final`关键字修饰的类在`javap`输出的`flags`中会体现为什么标志？

    A.`ACC_FINAL`

    B.`ACC_ABSTRACT`

    C.`ACC_SUPER`

    D.`ACC_PUBLIC`

22. 以下哪项是JVM中“垃圾回收”（Garbage Collection）的主要目标？

    A.将字节码转换为机器码。

    B.在程序运行时加载`.class`文件。

    C.优化CPU使用率。

    D.自动管理堆内存，回收不再被引用的对象。

23. 在JVM的内存区域中，哪一部分是用于存储方法运行时产生的局部变量和操作数栈的？

    A.方法区 (Method Area)

    B.堆 (Heap)

    C.程序计数器 (PC Register)

    D.Java虚拟机栈 (JVM Stack)

24. 在`javap`输出的常量池中，`Methodref`类型的条目通常指向哪两种其他类型的条目？

    A.`Fieldref`和`Methodref`

    B.`Utf8`和`Integer`

    C.`Class`和`NameAndType`

    D.`Class`和`String`

25. 如果一个Java方法在`javap`输出的`flags`中包含`ACC_STATIC`，这意味着什么？

    A.该方法是私有的，只能在类内部访问。

    B.该方法是一个类方法，不依赖于任何对象实例。

    C.该方法是同步的。

    D.该方法可以被子类重写。

26. 字节码指令`dup`的作用是什么？

    A.从局部变量表中复制一个值到另一个槽位。

    B.复制操作数栈顶的一个或多个字，并将其压回栈顶。

    C.将一个值从操作数栈复制到局部变量表。

    D.将操作数栈顶的两个值相加。

27. 方法描述符 `([Ljava/lang/String;)V` 最常用于表示哪个Java方法？

    A.`public String toString()`

    B.`public static void main(String[] args)`

    C.`public void run()`

    D.`public int hashCode()`

28. 在JVM中，当一个方法正常返回时，会发生什么？

    A.程序计数器重置为0。

    B.当前栈帧被销毁，并从Java虚拟机栈中弹出。

    C.当前线程被终止。

    D.垃圾回收器立即运行，清理所有不再使用的对象。

29. 以下哪种常量池条目类型用于表示一个类的全限定名？

    A.`Methodref`

    B.`Fieldref`

    C.`NameAndType`

    D.`Class`

30. 字节码指令`putstatic`的作用是什么？

    A.给对象实例字段赋值。

    B.将一个值从操作数栈存储到局部变量表。

    C.获取类的静态字段的值。

    D.给类的静态字段赋值。

31. 在Java中，当一个方法被`synchronized`关键字修饰时，在`javap`输出的`flags`中会体现为什么标志？

    A.`ACC_ABSTRACT`

    B.`ACC_SYNCHRONIZED`

    C.`ACC_FINAL`

    D.`ACC_PUBLIC`

32. JVM的类加载过程通常包括哪三个主要阶段？

    A.分析、优化、执行

    B.编译、链接、运行

    C.创建、初始化、销毁

    D.加载、验证、准备





> 答案：
>
> 1. A，`aload_0` 指令将局部变量表中索引为0的对象引用（在此上下文中是 `this`）压入操作数栈，为后续的 `invokespecial` 调用做准备。
>
> 2. B，JVM使用此属性以及 `LineNumberTable` 在异常堆栈跟踪中提供精确的调试信息，将字节码与源代码关联起来。为什么不选A？依赖关系通过常量池中的类和方法引用来管理，而不是 `SourceFile` 属性。
>
> 3. B：<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250604161349613.png" alt="image-20250604161349613" style="zoom:50%;" />
     >
     >    方法区用于存储已被虚拟机加载的类信息、常量、静态变量等数据，是所有线程共享的。
>
> 4. A，`this_class`通过索引指向常量池中的一个`CONSTANT_Class_info`结构，该结构进一步指向定义当前类名的UTF-8字符串。
>
> 5. B,如果程序员已经提供了至少一个构造器，编译器会认为程序员已经接管了对象创建的逻辑，因此不会再自动添加任何构造器。
>
> 6. A , 对于返回类型为`void`的方法（或构造器），`return`指令用于正常完成方法的执行，并导致当前栈帧从Java虚拟机栈中出栈。
>
> 7. D ,这个属性是调试和异常堆栈跟踪的关键，它允许工具（如调试器和JVM）将执行中的字节码位置与开发者编写的源代码行对应起来。
>
> 8. B ,程序计数器精确地指向下一条将要被解释器执行的字节码指令的地址，确保程序流程的正确执行。
>
> 9. D , 每当一个方法被调用，JVM就会为该方法创建一个新的栈帧，并将其压入当前线程的Java虚拟机栈中。
>
> 10. B , 首先int对应FieldType 中的字符中的`I`，其次String对应的是`Ljava/lang/String`
>
> 11. C， `iconst_1`是一条高效的指令，专门用于将`int`类型的常量值1加载到操作数栈顶。
>
> 12. D，`Utf8`条目用于存储各种字符串字面量，包括类名、方法名、字段名以及描述符等。
>
> 13. A，堆是JVM中最大的一块内存区域，所有线程共享，主要用于存储Java对象实例和数组，也是垃圾回收的主要区域。
>
> 14. A，`ACC_PUBLIC`标志表示该方法具有公共访问权限，可以被其他类调用。
>
> 15. C，`new`指令用于在堆上分配内存并创建新的对象实例，通常后面会跟`dup`和`invokespecial`来初始化对象。
>
> 16. A，`Ljava/lang/String;`代表`String`参数，`I`代表`int`参数，`V`代表`void`返回类型。
>
> 17. D，`this`引用允许实例方法在执行时访问和操作属于当前对象的字段和方法，它是实例方法能够工作的核心机制。
>
> 18. A，Java虚拟机栈是线程私有的，每个线程都有自己的栈，用于存储方法调用时的栈帧。
>
> 19. D，`NameAndType`条目将一个字段或方法的名称（UTF-8字符串）和其描述符（UTF-8字符串）关联起来，以便在`Fieldref`或`Methodref`中引用。
>
> 20. D，`getfield`指令用于从操作数栈顶的对象引用中，获取指定实例字段的值并将其压入操作数栈。
>
> 21. A，`ACC_FINAL`标志表示该类不能被继承，这与Java源代码中的`final`关键字修饰类是一致的。
>
> 22. D，垃圾回收器自动识别并回收堆中不再被程序引用的对象所占用的内存空间，从而防止内存泄漏和提高内存利用率。
>
> 23. D，Java虚拟机栈是线程私有的，每个方法调用都会创建一个栈帧，栈帧中包含局部变量表和操作数栈。
>
> 24. C , <img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20250604164928254.png" alt="image-20250604164928254" style="zoom:50%;" />
      >
      >     `Methodref`通过`Class`条目指定方法所属的类，通过`NameAndType`条目指定方法的名称和描述符。
>
> 25. B ,`ACC_STATIC`标志表示该方法是静态方法，可以直接通过类名调用，不需要创建对象实例。
>
> 26. B , `dup`指令用于复制操作数栈顶的一个或两个字（根据具体指令类型），并将其副本压入栈顶，常用于`new`指令之后来为构造器提供对象引用。
>
> 27. B ,这是Java应用程序的入口方法，它接受一个`String`数组作为参数，并且没有返回值，与描述符完全匹配。
>
> 28. B , 方法正常返回的典型操作是销毁并弹出其对应的栈帧，将控制权返回给调用者。
>
> 29. D ,`Class`类型的常量池条目用于表示一个类或接口的全限定名，它会指向一个`Utf8`条目来存储实际的名称字符串。
>
> 30. D , `putstatic`指令用于从操作数栈顶弹出一个值，并将其存储到指定的静态字段中
>
> 31. B , `ACC_SYNCHRONIZED`标志表示该方法是同步的，JVM会在方法进入时自动获取对象监视器，并在方法退出时释放。
>
> 32. D ,  类加载机制：加载，链接（验证，准备，解析），初始化；类的生命周期：加载，链接，初始化，使用，卸载。
>
> 











