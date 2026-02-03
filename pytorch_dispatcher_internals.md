# PyTorch Dispatcher 深度解析：运算符分发的核心引擎

## 核心问题：`torch.add(a, b)` 如何知道该调用哪段代码？

当你写 `torch.add(cpu_tensor, cpu_tensor)` 或 `torch.add(cuda_tensor, cuda_tensor)` 时，PyTorch 需要：

1. **识别设备** - 是 CPU 还是 CUDA？
2. **识别数据类型** - 是 float32 还是 int64？
3. **识别上下文** - 需要 autograd 吗？需要量化吗？
4. **选择内核** - 调用正确的底层实现

这一切都由 **Dispatcher（调度器）** 完成——PyTorch 最重要的内部机制之一。

---

## Dispatcher 是什么？

### 简化的心智模型

```python
# 你以为的 torch.add：
def add(a, b):
    return a + b  # 太简单了！

# 实际的 torch.add（伪代码）：
def add(a, b):
    # 1. 计算 dispatch key
    dispatch_key = compute_dispatch_key(a, b)
    
    # 2. 查找 dispatch table
    kernel = dispatch_table[dispatch_key]
    
    # 3. 调用对应的内核
    return kernel(a, b)
```

### Dispatch Table（分发表）

每个 PyTorch 算子都有一个分发表，存储不同场景的实现：

```
torch.add 的分发表：
┌─────────────────────┬─────────────────────────────┐
│ Dispatch Key        │ Function Pointer            │
├─────────────────────┼─────────────────────────────┤
│ CPU                 │ cpu_add_kernel()            │
│ CUDA                │ cuda_add_kernel()           │
│ AutogradCPU         │ autograd_cpu_add()          │
│ AutogradCUDA        │ autograd_cuda_add()         │
│ XLA                 │ xla_add_kernel()            │
│ Quantized           │ quantized_add_kernel()      │
│ Sparse              │ sparse_add_kernel()         │
│ BackendSelect       │ backend_select_fallback()   │
└─────────────────────┴─────────────────────────────┘
```

---

## Dispatch Key（分发键）的层次结构

### 核心概念

Dispatch Key 是一个**优先级排序的位掩码**，决定调用哪个内核。

```python
# 查看一个 Tensor 的 dispatch key set
import torch

# CPU tensor
cpu_tensor = torch.randn(3, 3)
print(torch._C._dispatch_key_set(cpu_tensor))
# 输出: DispatchKeySet(CPU, ADInplaceOrView, Autograd)

# CUDA tensor (requires_grad=True)
cuda_tensor = torch.randn(3, 3, device='cuda', requires_grad=True)
print(torch._C._dispatch_key_set(cuda_tensor))
# 输出: DispatchKeySet(CUDA, ADInplaceOrView, AutogradCUDA, AutocastCUDA)
```

### Dispatch Key 的优先级

```
高优先级（先处理）
    │
    ├─ Autograd          ← 自动求导
    ├─ BackendSelect     ← 后端选择
    ├─ Tracing           ← JIT 追踪
    ├─ Functionalize     ← 函数化
    │
    ├─ CUDA / CPU / XLA  ← 后端设备
    │
    └─ Math (fallback)   ← 通用数学实现
        │
低优先级（最后处理）
```

### 多个 Key 的组合

```python
# 一个 CUDA tensor (requires_grad=True) 的完整 key set：
keys = [
    'AutogradCUDA',    # ← 首先处理自动求导
    'CUDA',            # ← 然后执行 CUDA 内核
    'ADInplaceOrView', # ← 处理 in-place 操作
]

# 调用 torch.add(x, y) 时的执行流程：
# 1. 先调用 AutogradCUDA 的 add 实现（记录梯度）
# 2. AutogradCUDA 重新分发到 CUDA 的 add 实现
# 3. CUDA add 执行实际计算
```

---

## 完整示例：`torch.add` 的分发过程

### 代码级追踪

```cpp
// 1. 用户调用
at::Tensor result = at::add(x, y);

// 2. 进入 dispatcher
// 文件: build/aten/src/ATen/Functions.h (自动生成)
inline at::Tensor add(const at::Tensor& self, const at::Tensor& other) {
    static auto op = torch::Dispatcher::singleton()
        .findSchemaOrThrow("aten::add.Tensor", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, ...)>();
    
    return op.call(self, other, ...);  // ← 进入 dispatcher
}

// 3. Dispatcher 计算 dispatch key set
// 文件: c10/core/impl/TorchDispatchUtils.h
DispatchKeySet compute_dispatch_key_set(const TensorList& tensors) {
    DispatchKeySet key_set;
    
    // 合并所有 tensor 的 key
    for (const Tensor& t : tensors) {
        key_set = key_set | t.key_set();
    }
    
    // 添加全局 key（如 BackendSelect）
    key_set = key_set | global_key_set;
    
    return key_set;
}

// 4. 找到最高优先级的 key
// 假设是 AutogradCUDA
DispatchKey highest_key = key_set.highestPriorityTypeId();

// 5. 调用对应的内核
// 文件: torch/csrc/autograd/generated/VariableTypeEverything.cpp
at::Tensor autograd_cuda_add(
    c10::DispatchKeySet ks,
    const at::Tensor& self,
    const at::Tensor& other
) {
    // 记录 autograd 信息
    auto grad_fn = std::make_shared<AddBackward>();
    
    // 6. 排除 Autograd key，重新分发到 CUDA
    auto result = at::redispatch::add(
        ks.remove(DispatchKey::AutogradCUDA),  // ← 关键
        self,
        other
    );
    
    // 附加梯度函数
    attach_grad_fn(result, grad_fn);
    return result;
}

// 7. 最终到达 CUDA 内核
// 文件: build/aten/src/ATen/RegisterCUDA.cpp
at::Tensor cuda_add_kernel(const at::Tensor& self, const at::Tensor& other) {
    // 实际的 CUDA 计算
    return at::native::add_cuda(self, other);
}
```

### 可视化流程图

```
用户调用: torch.add(cuda_tensor, cuda_tensor)
    ↓
Dispatcher: 计算 DispatchKeySet
    DispatchKeySet = {AutogradCUDA, CUDA, ...}
    ↓
选择最高优先级 Key: AutogradCUDA
    ↓
调用 AutogradCUDA::add()
    - 记录梯度信息
    - 创建 AddBackward 节点
    ↓
重新分发（排除 AutogradCUDA）
    DispatchKeySet = {CUDA, ...}
    ↓
选择 CUDA Key
    ↓
调用 CUDA::add()
    - 启动 CUDA kernel
    - 在 GPU 上执行加法
    ↓
返回结果 Tensor
```

---

## 为什么需要 Dispatcher？

### 问题：一个算子，太多变体

```python
# torch.add 需要支持：
torch.add(cpu_float, cpu_float)         # CPU float32
torch.add(cuda_float, cuda_float)       # CUDA float32
torch.add(cpu_int, cpu_int)             # CPU int64
torch.add(quantized_tensor, ...)        # 量化
torch.add(sparse_tensor, ...)           # 稀疏
torch.add(xla_tensor, ...)              # XLA (TPU)
torch.add(mps_tensor, ...)              # Metal (Apple GPU)

# 每个组合需要不同的实现！
```

### 没有 Dispatcher 的噩梦

```cpp
// ❌ 没有 dispatcher 的代码（不可维护）
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.is_cuda()) {
        if (a.dtype() == torch::kFloat32) {
            if (a.requires_grad()) {
                // CUDA float32 with autograd
                return autograd_cuda_float32_add(a, b);
            } else {
                return cuda_float32_add(a, b);
            }
        } else if (a.dtype() == torch::kInt64) {
            // ...更多嵌套 if
        }
    } else if (a.is_cpu()) {
        // ...CPU 的所有变体
    } else if (a.is_xla()) {
        // ...XLA 的所有变体
    }
    // 😱 这会变成数千行的 if-else 迷宫
}
```

### 有了 Dispatcher 的优雅

```cpp
// ✅ 使用 dispatcher（模块化）

// 1. 注册 CPU 实现
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("add", cpu_add_kernel);
}

// 2. 注册 CUDA 实现
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("add", cuda_add_kernel);
}

// 3. 注册 Autograd 实现
TORCH_LIBRARY_IMPL(aten, AutogradCPU, m) {
    m.impl("add", autograd_cpu_add);
}

// Dispatcher 自动选择正确的实现！
```

---

## Dispatch Key 的完整列表

### 后端设备 Keys

```cpp
CPU           // x86/ARM CPU
CUDA          // NVIDIA GPU
XLA           // Google TPU
MPS           // Apple Metal GPU
HIP           // AMD GPU
FPGA          // FPGA 设备
IPU           // Graphcore IPU
PrivateUse1   // 自定义后端 1
PrivateUse2   // 自定义后端 2
PrivateUse3   // 自定义后端 3
```

### 功能性 Keys

```cpp
Autograd           // 自动求导（所有后端）
AutogradCPU        // CPU 自动求导
AutogradCUDA       // CUDA 自动求导
AutogradXLA        // XLA 自动求导

BackendSelect      // 后端选择（工厂函数）
Functionalize      // 函数化（移除 in-place）
Tracing            // JIT 追踪
Profiling          // 性能分析

ADInplaceOrView    // In-place 操作处理
Quantized          // 量化张量
Sparse             // 稀疏张量
```

### Alias Keys（别名键）

```cpp
CompositeExplicitAutograd  // 通用实现（需要手动 autograd）
CompositeImplicitAutograd  // 通用实现（自动 autograd）
```

---

## 实战案例：自定义 Dispatch 实现

### 案例 1：注册自定义算子

```cpp
// my_ops.cpp

#include <torch/torch.h>

// 1. 定义算子 schema
TORCH_LIBRARY(myops, m) {
    m.def("custom_add(Tensor self, Tensor other) -> Tensor");
}

// 2. 注册 CPU 实现
TORCH_LIBRARY_IMPL(myops, CPU, m) {
    m.impl("custom_add", [](const torch::Tensor& self, const torch::Tensor& other) {
        std::cout << "🖥️  使用 CPU 实现\n";
        return self + other;
    });
}

// 3. 注册 CUDA 实现
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
    m.impl("custom_add", [](const torch::Tensor& self, const torch::Tensor& other) {
        std::cout << "🚀 使用 CUDA 实现\n";
        return self.add(other);  // 调用 CUDA 加法
    });
}

// 4. 注册 Autograd 实现
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
    m.impl("custom_add", [](
        c10::DispatchKeySet ks,
        const torch::Tensor& self,
        const torch::Tensor& other
    ) {
        std::cout << "📊 处理 Autograd\n";
        
        // 重新分发到后端实现
        auto result = at::redispatch::custom_add(
            ks.remove(c10::DispatchKey::Autograd),
            self,
            other
        );
        
        // 附加梯度函数
        // ... (省略 autograd 代码)
        
        return result;
    });
}
```

### Python 调用

```python
import torch

# 编译和加载自定义算子
torch.ops.load_library("my_ops.so")

# CPU tensor
cpu_a = torch.randn(3, 3)
cpu_b = torch.randn(3, 3)
result = torch.ops.myops.custom_add(cpu_a, cpu_b)
# 输出: 🖥️  使用 CPU 实现

# CUDA tensor
cuda_a = torch.randn(3, 3, device='cuda')
cuda_b = torch.randn(3, 3, device='cuda')
result = torch.ops.myops.custom_add(cuda_a, cuda_b)
# 输出: 🚀 使用 CUDA 实现

# Autograd tensor
grad_a = torch.randn(3, 3, requires_grad=True)
grad_b = torch.randn(3, 3, requires_grad=True)
result = torch.ops.myops.custom_add(grad_a, grad_b)
# 输出: 📊 处理 Autograd
#       🖥️  使用 CPU 实现
```

---

## 高级主题：Boxed vs Unboxed Kernels

### Unboxed Kernels（默认）

```cpp
// 类型化的函数签名
Tensor cpu_add(const Tensor& self, const Tensor& other) {
    // 编译器知道确切的类型
    return self + other;
}
```

**优点**：
- 类型安全
- 性能最优（无装箱开销）

**缺点**：
- 每个算子需要单独编译

### Boxed Kernels（通用）

```cpp
// 通用的盒装签名
void boxed_add(const c10::OperatorHandle& op, c10::Stack* stack) {
    // 从栈中取参数
    auto self = pop(stack).toTensor();
    auto other = pop(stack).toTensor();
    
    // 计算结果
    auto result = self + other;
    
    // 推回栈
    push(stack, result);
}
```

**优点**：
- 一个实现支持所有算子
- 二进制文件更小

**缺点**：
- 装箱/拆箱有开销

### 何时使用

```python
# Unboxed: 核心算子（add, mul, conv）
# Boxed: Fallback 实现（如 CPU fallback）
```

---

## BackendSelect：工厂函数的特殊处理

### 问题：工厂函数没有 Tensor 参数

```python
# 没有输入 tensor，如何选择后端？
x = torch.zeros(3, 3, device='cuda')  # ❓ dispatcher 如何知道用 CUDA？
```

### 解决方案：BackendSelect Key

```cpp
// BackendSelect 的实现（简化）
Tensor zeros_backend_select(
    IntArrayRef size,
    optional<Device> device,
    ...
) {
    // 1. 从参数推断设备
    Device target_device = device.value_or(Device(kCPU));
    
    // 2. 选择对应的 dispatch key
    DispatchKey key;
    if (target_device.is_cuda()) {
        key = DispatchKey::CUDA;
    } else if (target_device.is_cpu()) {
        key = DispatchKey::CPU;
    }
    
    // 3. 直接分发到后端
    return at::redispatch::zeros(
        DispatchKeySet(key),
        size,
        ...
    );
}
```

---

## 性能优化：Dispatcher 的开销

### Benchmark

```python
import torch
import time

# 预热
x = torch.randn(1000, 1000, device='cuda')
for _ in range(100):
    _ = x + x

# 测试
iterations = 10000

# Dispatcher 开销
start = time.time()
for _ in range(iterations):
    _ = torch.add(x, x)
dispatch_time = time.time() - start

# 直接 CUDA kernel（假设可以绕过 dispatcher）
# 实际上无法绕过，这里是理论值
print(f"Dispatcher 调用: {dispatch_time:.4f}s")
print(f"平均每次: {dispatch_time / iterations * 1e6:.2f} μs")

# 典型输出：
# Dispatcher 调用: 0.0123s
# 平均每次: 1.23 μs
```

### 优化策略

#### 1. 内联小算子
```cpp
// 对于简单操作，dispatcher 可以内联
inline Tensor simple_add(const Tensor& a, const Tensor& b) {
    return a.add_(b);  // 绕过部分 dispatch 逻辑
}
```

#### 2. JIT 编译
```python
# TorchScript 可以优化掉部分 dispatch
@torch.jit.script
def fused_op(x, y):
    return (x + y) * 2  # 融合多个算子，减少 dispatch 次数
```

#### 3. torch.compile
```python
# PyTorch 2.0 的编译器
@torch.compile
def my_function(x, y):
    return x + y  # 编译期优化 dispatch
```

---

## 调试工具

### 1. 查看 Dispatch Table

```python
import torch

# 查看算子的所有注册实现
print(torch._C._dispatch_dump_table("aten::add.Tensor"))

# 输出示例：
# CPU: registered at ./build/aten/src/ATen/RegisterCPU.cpp:1309
# CUDA: registered at ./build/aten/src/ATen/RegisterCUDA.cpp:2420
# AutogradCPU: registered at ./torch/csrc/autograd/VariableType.cpp:523
# ...
```

### 2. Python Dispatcher（调试模式）

```python
from torch._python_dispatcher import PythonDispatcher

# 创建调试 dispatcher
dispatcher = PythonDispatcher()

# 注册自定义实现
@dispatcher.register('aten::add')
def debug_add(a, b):
    print(f"🔍 调用 add: {a.shape} + {b.shape}")
    return a + b

# 使用
with dispatcher:
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x + y  # 触发 debug_add
```

### 3. 跟踪 Dispatch 路径

```python
# 设置环境变量
import os
os.environ['TORCH_SHOW_DISPATCH_TRACE'] = '1'

# 运行代码
x = torch.randn(3, 3, device='cuda', requires_grad=True)
y = x + x

# 输出 dispatch 路径：
# Dispatching to: AutogradCUDA
# Redispatching to: CUDA
# Calling CUDA kernel
```

---

## 扩展 Dispatcher：添加自定义后端

### 完整示例：NPU 后端

```cpp
// 1. 定义 NPU 设备类型
// c10/core/DeviceType.h
enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
    // ...
    PrivateUse1 = 15,  // ← 使用这个槽位
};

// 2. 注册 dispatch key
// c10/core/DispatchKey.h
// PrivateUse1 已经预留

// 3. 实现 NPU kernels
// my_npu_backend.cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add", npu_add_kernel);
    m.impl("mul", npu_mul_kernel);
    // ... 更多算子
}

// 4. NPU 内核实现
Tensor npu_add_kernel(const Tensor& self, const Tensor& other) {
    // 调用 NPU 驱动
    npu_launch_add_kernel(
        self.data_ptr(),
        other.data_ptr(),
        self.numel()
    );
    return result;
}

// 5. 注册 Autograd
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("add", autograd_npu_add);
}
```

### Python 使用

```python
import torch

# 1. 加载 NPU 扩展
torch.ops.load_library("libtorch_npu.so")

# 2. 注册设备名称
torch.utils.rename_privateuse1_backend("npu")

# 3. 创建 NPU tensor
x = torch.randn(3, 3, device='npu')
y = torch.randn(3, 3, device='npu')

# 4. 自动 dispatch 到 NPU 实现
z = x + y  # 调用 npu_add_kernel
```

---

## 总结：Dispatcher 的设计哲学

1. **模块化** - 每个后端/功能独立实现
2. **可扩展** - 第三方可以轻松添加新后端
3. **高性能** - 间接调用开销 < 1 微秒
4. **类型安全** - C++ 模板保证编译期检查

### 为什么 Dispatcher 是 PyTorch 的核心？

```
没有 Dispatcher:
    ├─ 每个算子 = 几千行 if-else
    ├─ 添加新后端 = 修改所有算子
    └─ 无法支持第三方扩展

有了 Dispatcher:
    ├─ 每个算子 = 注册几个 kernel
    ├─ 添加新后端 = 实现 TORCH_LIBRARY_IMPL
    └─ 第三方可以无缝集成
```

**Dispatcher 让 PyTorch 从"一个框架"变成了"一个平台"。**

---

## 进阶资源

- **Edward Yang 博客**:
  - [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
  - [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/)

- **PyTorch 源码**:
  - `c10/core/DispatchKey.h` - Dispatch key 定义
  - `aten/src/ATen/core/dispatch/Dispatcher.h` - Dispatcher 主逻辑
  - `torch/csrc/autograd/generated/` - Autograd 代码生成

- **官方文档**:
  - [Dispatcher 教程](https://pytorch.org/tutorials/advanced/dispatcher.html)
  - [自定义算子](https://pytorch.org/tutorials/advanced/cpp_extension.html)

- **PyTorch Developer Podcast**:
  - Episode: "The Dispatcher"
  - Episode: "Boxed Kernels"
