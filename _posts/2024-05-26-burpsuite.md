---
layout: post
title: "Burpsuite"
subtitle: "Repeater模块无法返回response"
date: 2024-05-26
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
   - Burpsuite
---




# Burpsuite Repeater模块无法返回response

真是逆天了，今天下午打开很久没用的burp suite，但是抓包之后send to repeater，repeater模块无法返回response了：

![image-20240421151020698](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421151020698.png)

点了无数遍“send”，也重启了burp 都无法响应response。

如果burp放行，浏览器还会出现下面的提示：

![image-20240421152817414](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421152817414.png)

```
Burp Suite Professional
Error

No response received from remote server.
```

可能网站本身具有反 burp 的措施。

于是我又部署了另外一个靶场。还是一样的结果。这就证明是burp的配置出现了问题。

## 解决方法一：插件disable

![image-20240421153216429](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421153216429.png)

重启burp，并勾选如下两个选项 `Use Burp defaults` 、`Disable extensions`

这个方法对我没有用。

## 解决方法二：禁用`http/2`

取消勾选下面的`http/2`

![image-20240421153606341](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421153606341.png)

还是不行。

## 解决方法三：证书

这个问题是我自己发现的。因为前面的方法都不管用，于是我用fire fox的百度（之前一直用的Google）搜索了报错，出现了下面的报错：

![image-20240421154703126](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421154703126.png)

怀疑是证书过期。

去下载证书：

![image-20240421154858727](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421154858727.png)

选择一个合适的位置下载证书：

![image-20240421155023930](https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421155023930.png)

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421155043086.png" alt="image-20240421155043086" style="zoom:50%;" />

firefox---》设置---〉隐私与安全---》查看证书---〉导入，但是又报错：此证书已在此前安装为一个证书颁发机构：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240421155419974.png" alt="image-20240421155419974" style="zoom:50%;" />

可以尝试关闭浏览器，删除C:\Users\xx\AppData\Roaming\Mozilla\Firefox\Profiles\ufc3lps8.default\cert8.db

也可以更新firefox，重新导入证书。安装完成打开最新版，选择“创建新配置文件”。

重新配置插件foxy proxy

结果一顿操作还是不行。。。

于是我决定换浏览器。

在Chrome导入证书的时候需要注意，要将证书全部选为信任。不知道在firefox当中是不是因为没有选择信任菜不行。。。Chrome导入证书参考下面这篇博客：https://www.cnblogs.com/Hi-blog/p/How-To-Import-BurpSuite-Certificate-To-Chrome-On-MacOS.html

还是不行。

无语。换了一台ubuntu重新安装burp suite





