---
layout: post
title: "书籍阅读-Python灰帽子–黑客与逆向工程师的Python编程之道"
subtitle: "Python灰帽子–黑客与逆向工程师的Python编程之道"
date: 2024-05-26
author: "Hilda"
header-img: "img/post-bg-2015.jpg"
tags:
   - 书籍阅读
---

# <Python灰帽子–黑客与逆向工程师的Python编程之道>

完整的过程是：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240926105352242.png" alt="image-20240926105352242" style="zoom:50%;" />



## 1.启动可执行程序`CreateProcessA`

`CreateProcessA`是Windows API中的一个函数，用于创建一个新的进程并启动一个可执行文件。该函数允许你指定新进程的各种属性，如命令行参数、工作目录、环境变量等。`CreateProcessA`是`CreateProcess`函数的ANSI版本，用于处理ANSI字符串。

函数原型如下：

```c
BOOL CreateProcessA(
  LPCSTR                lpApplicationName,  // 可执行文件的路径
  LPSTR                 lpCommandLine,      // 命令行参数
  LPSECURITY_ATTRIBUTES lpProcessAttributes,// 进程安全属性
  LPSECURITY_ATTRIBUTES lpThreadAttributes, // 线程安全属性
  BOOL                  bInheritHandles,    // 是否继承句柄
  DWORD                 dwCreationFlags,    // 创建标志
  LPVOID                lpEnvironment,      // 环境变量
  LPCSTR                lpCurrentDirectory, // 当前目录
  LPSTARTUPINFOA        lpStartupInfo,      // 启动信息
  LPPROCESS_INFORMATION lpProcessInformation// 进程信息
);
```

书上提及的重要的参数是以下几个：

- `lpApplicationName`
- `lpCommandLine`
- `dwCreationFlags`
- `lpStartupInfo`
- `lpProcessInformation`



## 2.定义`Debugger`,`debugger_defines`和相关测试程序

因为操作系统的不兼容性，所以修改`my_debugger.py`，`my_debugger_defines.py`以及`my_test.py`内容如下：

`my_debugger.py`

```python
# -*- coding: utf-8 -*-
# @Date    : 2016-08-11 16:48:16
# @Author  : giantbranch (giantbranch@gmail.com)
# @Link    : http://blog.csdn.net/u012763794?viewmode=contents

from ctypes import *
import os
import signal
import sys
import time

libc = CDLL('libc.dylib')

# Define ptrace constants for macOS
PT_TRACE_ME = 0
PT_ATTACH = 10
PT_CONTINUE = 7
PT_READ_D = 2
PT_WRITE_D = 3
PT_DETACH = 11

class debugger():

    # 初始化
    # self参数表示类的实例
    def __init__(self):
        self.pid = None
        self.debugger_active = False
        self.breakpoints = {}
        self.first_breakpoint = True
        self.hardware_breakpoints = {}
        self.guarded_pages = []
        self.memory_breakpoints = {}

    # 启动程序
    def load(self, path_to_exe):
        pid = os.fork()
        if pid == 0:
            # Child process
            libc.ptrace(PT_TRACE_ME, 0, None, None)
            os.execv(path_to_exe, [path_to_exe])
        else:
            # Parent process
            self.pid = pid
            self.debugger_active = True
            print(f"[*] Process launched with PID: {pid}")

    # 附加
    def attach(self, pid):
        self.pid = pid
        libc.ptrace(PT_ATTACH, pid, None, None)
        self.debugger_active = True
        print(f"[*] Attached to process with PID: {pid}")

    # 运行
    def run(self):
        while self.debugger_active:
            self.get_debug_event()

    # 获取调试事件
    def get_debug_event(self):
        status = c_int()
        libc.waitpid(self.pid, byref(status), 0)
        if self.is_stopped(status.value):
            signal_number = self.get_stop_signal(status.value)
            if signal_number == signal.SIGTRAP:
                self.handle_breakpoint()
            elif signal_number == signal.SIGSEGV:
                print("Access Violation Detected.")
            else:
                print(f"Received signal: {signal_number}")
            libc.ptrace(PT_CONTINUE, self.pid, None, 0)

    # 处理断点
    def handle_breakpoint(self):
        if self.first_breakpoint:
            self.first_breakpoint = False
            print("[*] Hit the first breakpoint.")
        else:
            print("[*] Hit user defined breakpoint.")

    # 分离
    def detach(self):
        libc.ptrace(PT_DETACH, self.pid, None, None)
        self.debugger_active = False
        print("[*] Detached from process.")

    # 设置断点
    def bp_set(self, address):
        if address not in self.breakpoints:
            original_byte = self.read_process_memory(address, 1)
            self.write_process_memory(address, b'\xCC')
            self.breakpoints[address] = original_byte
        return True

    # 读内存
    def read_process_memory(self, address, length):
        data = b''
        for i in range(length):
            value = libc.ptrace(PT_READ_D, self.pid, c_void_p(address + i), 0)
            data += bytes([value])
        return data

    # 写内存
    def write_process_memory(self, address, data):
        for i, byte in enumerate(data):
            libc.ptrace(PT_WRITE_D, self.pid, c_void_p(address + i), byte)
        return True

    # 检查进程是否停止
    def is_stopped(self, status):
        return (status & 0x7f) == 0

    # 获取停止信号
    def get_stop_signal(self, status):
        return (status & 0x7f)

# 示例使用
if __name__ == "__main__":
    dbg = debugger()
    dbg.load("/System/Applications/Calculator.app/Contents/MacOS/Calculator")
    dbg.run()
```



``my_debugger_defines.py`

```python
# -*- coding: utf-8 -*-
# @Date    : 2016-08-11 16:07:38
# @Author  : giantbranch (giantbranch@gmail.com)
# @Link    : http://blog.csdn.net/u012763794?viewmode=contents

# 把所有的结构体，联合体，常量等放这，方便以后维护

from ctypes import *

# 给ctypes类型重新命名，跟windows编程接轨吧
BYTE = c_ubyte
WORD = c_ushort
DWORD = c_ulong
LPBYTE = POINTER(c_ubyte)
LPTSTR = POINTER(c_char)
HANDLE = c_void_p
PVOID = c_void_p
LPVOID = c_void_p
UINT_PTR = c_ulong
SIZE_T = c_ulong

# 常量
DEBUG_PROCESS = 0x00000001
CREATE_NEW_CONSOLE = 0x00000010
PROCESS_ALL_ACCESS = 0x001F0FFF
INFINITE = 0xFFFFFFFF
DBG_CONTINUE = 0x00010002

# 调试事件常量
EXCEPTION_DEBUG_EVENT = 0x1
CREATE_THREAD_DEBUG_EVENT = 0x2
CREATE_PROCESS_DEBUG_EVENT = 0x3
EXIT_THREAD_DEBUG_EVENT = 0x4
EXIT_PROCESS_DEBUG_EVENT = 0x5
LOAD_DLL_DEBUG_EVENT = 0x6
UNLOAD_DLL_DEBUG_EVENT = 0x7
OUTPUT_DEBUG_STRING_EVENT = 0x8
RIP_EVENT = 0x9

# 调试异常代号
EXCEPTION_ACCESS_VIOLATION = 0xC0000005
EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_GUARD_PAGE = 0x80000001
EXCEPTION_SINGLE_STEP = 0x80000004

# Thread constants for CreateToolhelp32Snapshot()
TH32CS_SNAPHEAPLIST = 0x00000001
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPTHREAD = 0x00000004
TH32CS_SNAPMODULE = 0x00000008
TH32CS_INHERIT = 0x80000000
TH32CS_SNAPALL = (TH32CS_SNAPHEAPLIST | TH32CS_SNAPPROCESS | TH32CS_SNAPTHREAD | TH32CS_SNAPMODULE)
THREAD_ALL_ACCESS = 0x001F03FF

# Context flags for GetThreadContext()
CONTEXT_FULL = 0x00010007
CONTEXT_DEBUG_REGISTERS = 0x00010010

# Memory permissions
PAGE_EXECUTE_READWRITE = 0x00000040

# Hardware breakpoint conditions
HW_ACCESS = 0x00000003
HW_EXECUTE = 0x00000000
HW_WRITE = 0x00000001

# Memory page permissions, used by VirtualProtect()
PAGE_NOACCESS = 0x00000001
PAGE_READONLY = 0x00000002
PAGE_READWRITE = 0x00000004
PAGE_WRITECOPY = 0x00000008
PAGE_EXECUTE = 0x00000010
PAGE_EXECUTE_READ = 0x00000020
PAGE_EXECUTE_READWRITE = 0x00000040
PAGE_EXECUTE_WRITECOPY = 0x00000080
PAGE_GUARD = 0x00000100
PAGE_NOCACHE = 0x00000200
PAGE_WRITECOMBINE = 0x00000400

# CreateProcessA()函数的结构,(用于设置创建子进程的各种属性)
class STARTUPINFO(Structure):
    _fields_ = [
        ("cb", DWORD),
        ("lpReserved", LPTSTR),
        ("lpDesktop", LPTSTR),
        ("lpTitle", LPTSTR),
        ("dwX", DWORD),
        ("dwY", DWORD),
        ("dwXSize", DWORD),
        ("dwYSize", DWORD),
        ("dwXCountChars", DWORD),
        ("dwYCountChars", DWORD),
        ("dwFillAttribute", DWORD),
        ("dwFlags", DWORD),
        ("wShowWindow", WORD),
        ("cbReserved2", WORD),
        ("lpReserved2", LPTSTR),
        ("hStdInput", DWORD),
        ("hStdOutput", DWORD),
        ("hStdError", DWORD),
    ]

# 进程的信息：进程线程的句柄，进程线程的id
class PROCESS_INFORMATION(Structure):
    _fields_ = [
        ("hProcess", HANDLE),
        ("hThread", HANDLE),
        ("dwProcessId", DWORD),
        ("dwThreadId", DWORD),
    ]

# When the dwDebugEventCode is evaluated
class EXCEPTION_RECORD(Structure):
    pass

EXCEPTION_RECORD._fields_ = [
    ("ExceptionCode", DWORD),
    ("ExceptionFlags", DWORD),
    ("ExceptionRecord", POINTER(EXCEPTION_RECORD)),
    ("ExceptionAddress", PVOID),
    ("NumberParameters", DWORD),
    ("ExceptionInformation", UINT_PTR * 15),
]

class _EXCEPTION_RECORD(Structure):
    _fields_ = [
        ("ExceptionCode", DWORD),
        ("ExceptionFlags", DWORD),
        ("ExceptionRecord", POINTER(EXCEPTION_RECORD)),
        ("ExceptionAddress", PVOID),
        ("NumberParameters", DWORD),
        ("ExceptionInformation", UINT_PTR * 15),
    ]

# Exceptions
class EXCEPTION_DEBUG_INFO(Structure):
    _fields_ = [
        ("ExceptionRecord", EXCEPTION_RECORD),
        ("dwFirstChance", DWORD),
    ]

# it populates this union appropriately
class DEBUG_EVENT_UNION(Union):
    _fields_ = [
        ("Exception", EXCEPTION_DEBUG_INFO),
        #        ("CreateThread",      CREATE_THREAD_DEBUG_INFO),
        #        ("CreateProcessInfo", CREATE_PROCESS_DEBUG_INFO),
        #        ("ExitThread",        EXIT_THREAD_DEBUG_INFO),
        #        ("ExitProcess",       EXIT_PROCESS_DEBUG_INFO),
        #        ("LoadDll",           LOAD_DLL_DEBUG_INFO),
        #        ("UnloadDll",         UNLOAD_DLL_DEBUG_INFO),
        #        ("DebugString",       OUTPUT_DEBUG_STRING_INFO),
        #        ("RipInfo",           RIP_INFO),
    ]

# DEBUG_EVENT describes a debugging event
# that the debugger has trapped
class DEBUG_EVENT(Structure):
    _fields_ = [
        ("dwDebugEventCode", DWORD),
        ("dwProcessId", DWORD),
        ("dwThreadId", DWORD),
        ("u", DEBUG_EVENT_UNION),
    ]

# Used by the CONTEXT structure
class FLOATING_SAVE_AREA(Structure):
    _fields_ = [

        ("ControlWord", DWORD),
        ("StatusWord", DWORD),
        ("TagWord", DWORD),
        ("ErrorOffset", DWORD),
        ("ErrorSelector", DWORD),
        ("DataOffset", DWORD),
        ("DataSelector", DWORD),
        ("RegisterArea", BYTE * 80),
        ("Cr0NpxState", DWORD),
    ]

# The CONTEXT structure which holds all of the
# register values after a GetThreadContext() call
class CONTEXT(Structure):
    _fields_ = [

        ("ContextFlags", DWORD),
        ("Dr0", DWORD),
        ("Dr1", DWORD),
        ("Dr2", DWORD),
        ("Dr3", DWORD),
        ("Dr6", DWORD),
        ("Dr7", DWORD),
        ("FloatSave", FLOATING_SAVE_AREA),
        ("SegGs", DWORD),
        ("SegFs", DWORD),
        ("SegEs", DWORD),
        ("SegDs", DWORD),
        ("Edi", DWORD),
        ("Esi", DWORD),
        ("Ebx", DWORD),
        ("Edx", DWORD),
        ("Ecx", DWORD),
        ("Eax", DWORD),
        ("Ebp", DWORD),
        ("Eip", DWORD),
        ("SegCs", DWORD),
        ("EFlags", DWORD),
        ("Esp", DWORD),
        ("SegSs", DWORD),
        ("ExtendedRegisters", BYTE * 512),
    ]

# THREADENTRY32 contains information about a thread
# we use this for enumerating all of the system threads

class THREADENTRY32(Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("cntUsage", DWORD),
        ("th32ThreadID", DWORD),
        ("th32OwnerProcessID", DWORD),
        ("tpBasePri", DWORD),
        ("tpDeltaPri", DWORD),
        ("dwFlags", DWORD),
    ]

# Supporting struct for the SYSTEM_INFO_UNION union
class PROC_STRUCT(Structure):
    _fields_ = [
        ("wProcessorArchitecture", WORD),
        ("wReserved", WORD),
    ]

# Supporting union for the SYSTEM_INFO struct
class SYSTEM_INFO_UNION(Union):
    _fields_ = [
        ("dwOemId", DWORD),
        ("sProcStruc", PROC_STRUCT),
    ]

# SYSTEM_INFO structure is populated when a call to
# kernel32.GetSystemInfo() is made. We use the dwPageSize
# member for size calculations when setting memory breakpoints
class SYSTEM_INFO(Structure):
    _fields_ = [
        ("uSysInfo", SYSTEM_INFO_UNION),
        ("dwPageSize", DWORD),
        ("lpMinimumApplicationAddress", LPVOID),
        ("lpMaximumApplicationAddress", LPVOID),
        ("dwActiveProcessorMask", DWORD),
        ("dwNumberOfProcessors", DWORD),
        ("dwProcessorType", DWORD),
        ("dwAllocationGranularity", DWORD),
        ("wProcessorLevel", WORD),
        ("wProcessorRevision", WORD),
    ]

# MEMORY_BASIC_INFORMATION contains information about a
# particular region of memory. A call to kernel32.VirtualQuery()
# populates this structure.
class MEMORY_BASIC_INFORMATION(Structure):
    _fields_ = [
        ("BaseAddress", PVOID),
        ("AllocationBase", PVOID),
        ("AllocationProtect", DWORD),
        ("RegionSize", SIZE_T),
        ("State", DWORD),
        ("Protect", DWORD),
        ("Type", DWORD),
    ]
```







`my_test.py`

```python
# -*- coding: utf-8 -*-
# @Date    : 2016-08-12 14:18:10
# @Author  : giantbranch (giantbranch@gmail.com)
# @Link    : http://blog.csdn.net/u012763794?viewmode=contents

# use: python my_test.py
import my_debugger
from my_debugger_defines import *
debugger = my_debugger.debugger()
debugger.load("/System/Applications/Calculator.app/Contents/MacOS/Calculator")

```

> 注：
>
> 1.在macOS上，调试器和系统调用的API与Windows有所不同。macOS使用POSIX标准和Mach内核提供的API。由于macOS没有直接等同于Windows的调试API，将使用`ptrace`和`sysctl`等POSIX和Mach内核API来实现类似的功能。
>
> 2.
>
> 1. **加载和附加进程**：
     >    - `load`方法使用`os.fork`和`os.execv`来启动新进程，并使用`ptrace`来调试子进程。
>    - `attach`方法使用`ptrace`来附加到现有进程。
> 2. **调试事件处理**：
     >    - `get_debug_event`方法使用`waitpid`来等待子进程的状态变化，并根据信号类型处理调试事件。
>    - `handle_breakpoint`方法处理断点事件。
> 3. **内存操作**：
     >    - `read_process_memory`和`write_process_memory`方法使用`ptrace`来读写目标进程的内存。
> 4. **断点设置**：
     >    - `bp_set`方法在指定地址设置断点，并将原始字节保存以便恢复。
>
> 3.在macOS上，`ptrace`函数的调用方式与Linux有所不同。macOS的`ptrace`函数需要通过`libc`库来调用，并且需要使用正确的常量。`PT_TRACE_ME`是Linux上的常量，在macOS上应该使用`PT_ATTACH`等常量。
>
> 除此之外还需修改：
>
> 1. **定义常量**：在macOS上，`ptrace`的常量与Linux不同。我们定义了`PT_TRACE_ME`、`PT_ATTACH`、`PT_CONTINUE`、`PT_READ_D`、`PT_WRITE_D`和`PT_DETACH`等常量。
> 2. **加载和附加进程**：
     >    - `load`方法使用`os.fork`和`os.execv`来启动新进程，并使用`ptrace`来调试子进程。
>    - `attach`方法使用`ptrace`来附加到现有进程。
> 3. **调试事件处理**：
     >    - `get_debug_event`方法使用`waitpid`来等待子进程的状态变化，并根据信号类型处理调试事件。
>    - `handle_breakpoint`方法处理断点事件。
> 4. **内存操作**：
     >    - `read_process_memory`和`write_process_memory`方法使用`ptrace`来读写目标进程的内存。
> 5. **断点设置**：
     >    - `bp_set`方法在指定地址设置断点，并将原始字节保存以便恢复。
> 6. **检查进程是否停止**：
     >    - `is_stopped`方法检查进程是否停止。
> 7. **获取停止信号**：
     >    - `get_stop_signal`方法获取停止信号。
>
> 4.`/System/Applications/Calculator.app`不是可执行的，应该改成`/System/Applications/Calculator.app/Contents/MacOS/Calculator`

运行结果：

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240924100730774.png" alt="image-20240924100730774" style="zoom:50%;" />

而且每次运行，`PID`号不同。

## 3.获取“句柄”

> OS学的不精，补充一下句柄：
>
> 在计算机编程中，“句柄”是一个抽象引用，通常用于表示系统资源，比如文件、窗口或进程。在操作系统中，获取一个句柄允许程序访问和操作这些资源。
>
> 获取句柄的具体意义：
>
> 1. **句柄的作用**：句柄使得程序可以通过一个简化的标识符与资源进行交互，而不需要直接处理资源的底层细节。
> 2. **资源管理**：操作系统使用句柄来跟踪和管理资源，以确保它们在使用时不会被意外释放或修改。
>
>
>
> - 假设在开发一个Windows应用程序，想要监控一个特定的进程（比如一个游戏）。你可以使用`kernel32.dll`中的`OpenProcess`函数来获取该进程的句柄。
    >
    >   - OpenProcess的函数原型如下：若这个函数成功就会返回一个指向目标进程对象的句柄。
>
>   - ```cpp
>     HANDLE OpenProcess(
>       DWORD dwDesiredAccess,
>       BOOL bInheritHandle,
>       DWORD dwProcessId
>     );
>     ```
>
>   - 参数说明：
      >
      >     - **dwDesiredAccess**：请求的访问权限，例如`PROCESS_ALL_ACCESS`表示请求所有权限。在这本书中，这个参数作者设为`PROCESS_ALL_ACCESS`,是为了满足调试的需求。
>     - **bInheritHandle**：指定是否允许子进程继承该句柄。书中设置为`false`。
>     - **dwProcessId**：目标进程的ID。
>
> - 可以使用`libproc`库来获取进程信息，`libproc`提供了一些与进程相关的功能。要实现类似于`OpenProcess`的功能，通常你会用到`proc_pidinfo`函数来获取进程信息。
>
> - **特别注意：macOS或者Linux 无法获取进程的句柄！！！**原因：macOS和Linux的进程管理与Windows有所不同。它们不使用“句柄”的概念，而是通过进程ID（PID）来标识和操作进程。
    >
    >   - 主要区别：
          >     - **Windows**：使用句柄（如`OpenProcess`）来引用和管理进程，句柄可以直接用于系统调用。
>     - **macOS/Linux**：通过进程ID进行管理，通常使用`proc_pidinfo`、`ptrace`等系统调用来与进程交互。

类似于`kernel32.dll`的`OpenProcess`，`libc`的`proc_pidinfo`用于获取进程信息。

Debugger中定义的：

```python
    # Mac:获取进程的句柄，要调试当然要全不权限了
    def open_process(self, pid: int) -> c_void_p:
        """获取进程的伪句柄."""
        info = ProcBSDInfo()

        # 调用proc_pidinfo
        ret = libc.proc_pidinfo(pid, 1, 0, byref(info), sizeof(info))

        if ret > 0:
            print(f"成功获取进程信息: 进程ID: {info.pbi_pid}, 进程名: {info.pbi_comm.decode('utf-8')}")
            self.h_process = c_void_p(pid)  # 这里返回进程ID作为伪句柄
            return self.h_process
        else:
            print(f"无法获取进程 {pid} 的信息。")
            return None

```

需要先定义类`ProcBSDInfo`:

```python
# 定义PROC_PIDTASKINFO结构，用于open_process函数
class ProcBSDInfo(Structure):
    _fields_ = [
        ('pbi_pid', c_int),           # 进程ID
        ('pbi_comm', c_char * 256),  # 进程名称
        # 可以根据需要添加更多字段
    ]
```



> 注：在macOS中，获取“句柄”的概念与Windows不同。虽然你可以使用`proc_pidinfo`获取进程信息，但并没有直接的“句柄”概念。
>
> 在上面的示例中，`open_process`方法返回的是一个与进程ID相关的`c_void_p`类型，这可以视为一个“句柄”。然而，它并不是像Windows中的句柄那样能够直接用于进程管理。
>
> 如果你需要对进程进行进一步的操作，如读写内存或调试，macOS通常使用`ptrace`等系统调用，这些调用允许你控制和监视其他进程。
>
> 【模拟句柄的示例】
>
> 在示例代码中，`self.h_process = c_void_p(pid)`相当于返回了一个代表进程的“句柄”，但请记住：
>
> - **并不是一个真正的句柄**：它只是一个指向进程ID的指针，并不能直接用于系统调用。
> - **后续操作**：如果需要执行更复杂的操作，可能需要使用其他系统调用（如`ptrace`）来实现具体功能。

## 4.进程的附加

`DebugActiveProcess`是Windows API中的一个函数，用于将调试器附加到一个正在运行的进程上。这个函数允许调试器开始调试一个已经存在的进程，而不是创建一个新的进程。

函数原型如下：

```c
BOOL DebugActiveProcess(
  DWORD dwProcessId  // 要附加的进程的ID
);
```

在macOS上，类似于`DebugActiveProcess`的操作可以通过`ptrace`函数实现。

```python
PT_ATTACH = 10
def attach(self, pid):
    self.pid = pid
    libc.ptrace(PT_ATTACH, pid, None, None)
    self.debugger_active = True
    print(f"[*] Attached to process with PID: {pid}")
```

> 其中，`ptrace`的函数原型是:
>
> `long ptrace(int request, pid_t pid, void *addr, void *data);`
>
> - **`PT_ATTACH = 10`**：这是一个常量，用于表示`ptrace`系统调用中的`PT_ATTACH`命令。该命令用于附加到指定的进程，允许调用进程控制和监视被附加的进程。
    >
    >   - 注意：`my_debuggers.py`中一开始有`ptrace`的常量
>
>   - ```python
>     # Define ptrace constants for macOS
>     PT_TRACE_ME = 0
>     PT_ATTACH = 10
>     PT_CONTINUE = 7
>     PT_READ_D = 2
>     PT_WRITE_D = 3
>     PT_DETACH = 11
>     ```
>
>   - **`PT_TRACE_ME = 0`**
      >
      >     - 这个常量用于让当前进程告诉内核自己要被调试。调用这个命令后，内核会允许调试器对这个进程进行监控。
      >
      >     **`PT_ATTACH = 10`**
      >
      >     - 该命令用于将调试器附加到一个已经运行的进程。使用这个命令后，调试器可以控制被附加的进程，并获取其状态信息。
      >
      >     **`PT_CONTINUE = 7`**
      >
      >     - 这个常量用于继续一个被调试的进程的执行。在调试过程中，进程可能会因为某些事件（如断点）被暂停，使用这个命令可以使其恢复运行。
      >
      >     **`PT_READ_D = 2`**
      >
      >     - 这个命令用于从被调试进程的地址空间中读取数据。调用这个命令后，可以获取指定地址的数据。
      >
      >     **`PT_WRITE_D = 3`**
      >
      >     - 这个常量用于向被调试进程的地址空间写入数据。使用此命令后，可以修改被调试进程内存中指定地址的内容。
      >
      >     **`PT_DETACH = 11`**
      >
      >     - 这个命令用于从被调试进程中分离调试器。使用后，进程将继续正常运行，而调试器不再控制它。
>
> - `pid`: 被附加进程的进程ID。这个进程将被调试。
>
> - `None, None`: 这些参数在附加时通常不需要使用，可以用`None`来表示。

**`self.debugger_active = True`**: 将类的实例变量`self.debugger_active`设置为`True`，表示调试器现在处于活动状态，已成功附加到进程。

## 5.等待和处理调试事件

`WaitForDebugEvent` 是 Windows API 中用于调试的函数，它会在调试事件发生时阻塞当前线程，并返回相关的调试事件信息

下面是`WaitForDebugEvent`的函数原型：

```c
BOOL WaitForDebugEvent(
  LPDEBUG_EVENT lpDebugEvent,//指向 DEBUG_EVENT 结构体的指针，用于接收调试事件信息
  DWORD         dwMilliseconds // 等待的超时时间，单位为毫秒
);
```

`libc`中是`libc.waitpid(self.pid, byref(status), 0)`函数有类似的功能。函数原型如下：

```c
#include <sys/types.h>
#include <sys/wait.h>

pid_t waitpid(pid_t pid, int *status, int options);
```

执行的结果会存储在`status`中。

完整代码：

```c
# 获取调试事件
def get_debug_event(self):
    #  创建一个c_int类型的变量status，用于存储waitpid函数返回的进程状态信息
    status = c_int()
    # 调用waitpid函数，等待指定进程（通过self.pid）的状态变化，并将结果存储在status中。参数0表示阻塞，直到有状态变化发生。
    libc.waitpid(self.pid, byref(status), 0)
    # 检查进程是否因信号停止
    if self.is_stopped(status.value):
        signal_number = self.get_stop_signal(status.value)
        if signal_number == signal.SIGTRAP:
            self.handle_breakpoint()
        elif signal_number == signal.SIGSEGV:
            print("Access Violation Detected.")
        else:
            print(f"Received signal: {signal_number}")
        libc.ptrace(PT_CONTINUE, self.pid, None, 0)
```

## 6.恢复

`kernel32.ContinueDebugEvent` 是 Windows API 中的一个函数，用于在调试器接收到调试事件后，指示操作系统继续执行被调试的进程或线程。这个函数通常在调试器处理完调试事件（如断点、异常等）后调用，以便让被调试的进程或线程继续执行。

函数原型：

```c
BOOL ContinueDebugEvent(
  DWORD dwProcessId,
  DWORD dwThreadId,
  DWORD dwContinueStatus
);
```

1. **dwProcessId**
    - **类型**: `DWORD`
    - **描述**: 被调试进程的进程 ID。这个 ID 通常是从调试事件结构体（如 `DEBUG_EVENT`）中获取的。
2. **dwThreadId**
    - **类型**: `DWORD`
    - **描述**: 被调试线程的线程 ID。这个 ID 通常也是从调试事件结构体（如 `DEBUG_EVENT`）中获取的。
3. **dwContinueStatus**
    - **类型**: `DWORD`
    - **描述**: 指示操作系统如何继续执行被调试的线程。这个参数可以是以下两个值之一：
        - `DBG_CONTINUE`: 指示操作系统继续执行线程，并忽略异常（通常用于处理已处理的异常）。
        - `DBG_EXCEPTION_NOT_HANDLED`: 指示操作系统继续执行线程，但将异常传递给线程的默认异常处理程序（通常用于未处理的异常）。

`libc.ptrace(PT_CONTINUE, self.pid, None, 0)`可以实现类似的功能

## 7.分离调试器和被调试程序

这是附加进程之后需要进行的。

`winAPI`是`kernel32.DebugActiveProcessStop`,函数原型如下：

```c
BOOL DebugActiveProcessStop(
  DWORD dwProcessId
);
```

只需传入进程ID 即可。

同样在Mac中，用`libc.ptrace`实现:`libc.ptrace(PT_DETACH, self.pid, None, None)`

## 8.测试

测试：

```python
# -*- coding: utf-8 -*-
# @Date    : 2016-08-12 14:18:10
# @Author  : giantbranch (giantbranch@gmail.com)
# @Link    : http://blog.csdn.net/u012763794?viewmode=contents

# use: python my_test.py
import my_debugger
from my_debugger_defines import *

debugger = my_debugger.debugger()

# macOS 上的计算器应用程序路径
calculator_path = "/System/Applications/Calculator.app/Contents/MacOS/Calculator"

# 加载计算器应用程序
debugger.load(calculator_path)

# 获取用户输入的 PID
pid = input("Enter the PID of the process to attach to:")
debugger.attach(int(pid))

debugger.detach()
```

从活动监视器可以得到`Calculator`的PID是5167

<img src="https://wechat01.oss-cn-hangzhou.aliyuncs.com/img/image-20240926111454500.png" alt="image-20240926111454500" style="zoom:50%;" />

