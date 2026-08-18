# source-pytorch

> PyTorch 源码机制深度剖析文档集：Dispatcher 内幕、`nn.Module` 调用机制与 `module.apply` 机制。

## 📖 项目简介

针对 PyTorch **源码内部机制**的技术剖析仓库，以 Markdown 文档形式解析框架底层的工作原理——从 dispatcher 分发系统，到 `nn.Module` 的调用链与 `apply` 递归机制。

## 📚 文档

| 文档 | 剖析对象 |
|------|----------|
| `pytorch_dispatcher_internals.md` | PyTorch Dispatcher 内幕：分发、多后端、算子注册 |
| `nn_module_call_mechanism.md` | `nn.Module.__call__` 机制：hook、forward 与自动求导接线 |
| `module_apply_mechanism.md` | `module.apply()` 递归机制：如何逐层应用到子模块 |

## 🚀 使用

```bash
open pytorch_dispatcher_internals.md    # 从 Dispatcher 开始
```

## 📌 定位

面向想深入理解 PyTorch 内部实现的工程师，文档型源码笔记，随源码演进持续补充。