# BatchNorm作用验证实验 - 改进版 V2

全面监控BatchNorm的影响：激活值分布、权重范数、梯度范数

## 🎯 实验目的

本实验从**三个维度**全面验证BatchNorm的作用：

### 1. 激活值分布监控
验证BatchNorm是否使层输入分布更规范：
- ✅ 均值接近0
- ✅ 标准差接近1  
- ✅ 偏度接近0（对称分布）
- ✅ 峰度接近0（正态分布）

### 2. 权重范数监控 ⭐ NEW
观察权重在训练过程中的演变：
- 📊 **L1范数**：权重的绝对值之和，衡量稀疏性
- 📊 **L2范数**：权重的欧几里得范数，衡量整体大小
- 📊 **最大范数**：权重的最大绝对值
- 📊 **谱范数**：权重矩阵的最大奇异值

### 3. 梯度范数监控 ⭐ NEW
检测梯度消失/爆炸问题：
- 🔍 梯度L2范数过小 → 梯度消失
- 🔍 梯度L2范数过大 → 梯度爆炸
- 🔍 BatchNorm应使梯度更稳定

## 📁 文件说明

- **batchnorm_experiment_v2.py**: Python脚本版本（可本地运行）
- **batchnorm_experiment_v2.ipynb**: Jupyter Notebook版本（推荐用于Colab）
- **README_v2.md**: 本文档

## 🚀 快速开始

### 在Google Colab中运行（推荐）

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `batchnorm_experiment_v2.ipynb`
3. 确保使用GPU：Runtime → Change runtime type → GPU
4. 按顺序执行所有单元格
5. 在内嵌的TensorBoard中实时查看结果

### 本地运行

```bash
# 安装依赖
pip install torch torchvision scipy tensorboard

# 运行实验
python batchnorm_experiment_v2.py

# 在另一个终端启动TensorBoard
tensorboard --logdir=runs/batchnorm_experiment_v2

# 在浏览器打开
# http://localhost:6006
```

## 📊 TensorBoard可视化指南

### SCALARS Tab（标量指标）

#### 1. 性能指标对比
```
带BatchNorm/Loss/train          vs  不带BatchNorm/Loss/train
带BatchNorm/Accuracy/test       vs  不带BatchNorm/Accuracy/test
```

#### 2. 权重范数演变 ⭐ 重点
```
WithBN/WeightNorms/conv1/l2_norm     vs  WithoutBN/WeightNorms/conv1/l2_norm
WithBN/WeightNorms/conv1/l1_norm     vs  WithoutBN/WeightNorms/conv1/l1_norm
WithBN/WeightNorms/conv1/max_norm    vs  WithoutBN/WeightNorms/conv1/max_norm
WithBN/WeightNorms/conv1/spectral_norm vs WithoutBN/WeightNorms/conv1/spectral_norm
```

**观察要点**:
- 📈 **L2范数趋势**: 
  - 带BN: 应该逐渐稳定，不会无限增长
  - 不带BN: 可能持续增长或剧烈波动
  
- 📉 **权重变化速率**:
  - 带BN: 权重更新更稳定
  - 不带BN: 可能出现突变

#### 3. 梯度范数监控 ⭐ 重点
```
WithBN/GradNorms/conv1/l2_norm    vs  WithoutBN/GradNorms/conv1/l2_norm
WithBN/GradNorms/fc1/max_norm     vs  WithoutBN/GradNorms/fc1/max_norm
```

**观察要点**:
- 🔍 **梯度范数大小**:
  - 过小（< 1e-7）→ 梯度消失
  - 过大（> 100）→ 梯度爆炸
  - BatchNorm应使梯度保持在健康范围

- 🔍 **梯度稳定性**:
  - 带BN: 梯度范数相对稳定
  - 不带BN: 可能剧烈波动

#### 4. 激活值分布统计
```
WithBN/Activations/bn1/mean       vs  WithoutBN/Activations/conv1/mean
WithBN/Activations/bn1/std        vs  WithoutBN/Activations/conv1/std
WithBN/Activations/bn1/skewness   vs  WithoutBN/Activations/conv1/skewness
```

### HISTOGRAMS Tab（分布直方图）

#### 1. 权重分布演变
```
WithBN/Weights/conv1    vs  WithoutBN/Weights/conv1
WithBN/Weights/fc1      vs  WithoutBN/Weights/fc1
```

**观察要点**:
- 观察权重分布的形状变化
- BatchNorm后的权重分布应更规则

#### 2. 梯度分布演变
```
WithBN/Gradients/conv1  vs  WithoutBN/Gradients/conv1
```

**观察要点**:
- 梯度分布不应过于集中在0（梯度消失）
- 梯度分布不应有极端值（梯度爆炸）

## 🔬 实验配置

### 数据集
- **CIFAR-10**: 60,000张32×32彩色图像
- 10个类别
- 训练集: 50,000 | 测试集: 10,000

### 模型架构
```
SimpleCNN / SimpleCNN_BN:
├── Conv1 (3→64) + [BN] + ReLU + MaxPool
├── Conv2 (64→128) + [BN] + ReLU + MaxPool
├── Conv3 (128→256) + [BN] + ReLU + MaxPool
├── FC1 (4096→512) + [BN] + ReLU + Dropout(0.3)
└── FC2 (512→10)
```

### 训练参数
- **Batch Size**: 128
- **Epochs**: 15
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: StepLR (step_size=5, gamma=0.5)
- **Loss**: CrossEntropyLoss

### 监控的层

#### 激活值监控
- 带BN模型: `bn1`, `bn2`, `bn3`, `bn_fc`
- 不带BN模型: `conv1`, `conv2`, `conv3`, `fc1`

#### 权重范数监控
- 两个模型: `conv1`, `conv2`, `conv3`, `fc1`, `fc2`

## 📈 预期实验结果

### ✅ 带BatchNorm的模型

| 指标 | 预期表现 |
|-----|---------|
| **权重L2范数** | 逐渐稳定，不会无限增长 |
| **权重变化率** | 平滑稳定 |
| **梯度L2范数** | 保持在健康范围（1e-4 ~ 10） |
| **梯度稳定性** | 波动较小 |
| **激活值均值** | 接近0 |
| **激活值标准差** | 接近1 |
| **激活值偏度** | 接近0 |
| **训练收敛** | 更快 |
| **最终准确率** | 更高 |

### ❌ 不带BatchNorm的模型

| 指标 | 可能表现 |
|-----|---------|
| **权重L2范数** | 持续增长或剧烈波动 |
| **权重变化率** | 不稳定，可能突变 |
| **梯度L2范数** | 可能过小（消失）或过大（爆炸） |
| **梯度稳定性** | 波动较大 |
| **激活值均值** | 偏离0 |
| **激活值标准差** | 偏离1 |
| **激活值偏度** | 较大（分布不对称） |
| **训练收敛** | 较慢 |
| **最终准确率** | 较低 |

## 🔍 关键发现解释

### 1. 为什么权重范数很重要？

**权重范数反映了模型的容量和泛化能力**:

- **L2范数持续增长** → 可能过拟合
- **L2范数剧烈波动** → 训练不稳定
- **最大范数过大** → 某些权重过于dominant
- **BatchNorm通过规范化激活值，间接稳定了权重更新**

### 2. 为什么梯度范数很重要？

**梯度范数直接影响训练效率**:

```
权重更新: w_new = w_old - lr * gradient
```

- **梯度过小** → 权重几乎不更新 → 学习停滞
- **梯度过大** → 权重剧烈变化 → 训练发散
- **BatchNorm通过减少内部协变量偏移，使梯度更稳定**

### 3. BatchNorm如何稳定权重？

```
没有BN的情况:
输入分布变化 → 激活值分布剧变 → 梯度不稳定 → 权重更新不稳定

有BN的情况:
输入分布变化 → BN规范化 → 激活值分布稳定 → 梯度稳定 → 权重更新稳定
```

## 💡 TensorBoard使用技巧

### 多指标对比
1. 在SCALARS左侧勾选要对比的指标
2. 点击"Show data download links"下载数据
3. 使用"Toggle all runs"快速切换

### 平滑曲线
- 调整左侧的"Smoothing"滑块（0.0-0.999）
- 推荐值: 0.6-0.8

### 时间轴对齐
- 选择"Align by step"（按epoch对齐）
- 或"Align by wall time"（按实际时间对齐）

### 导出图表
- 点击右上角的下载按钮
- 支持PNG和SVG格式

## 🧪 扩展实验

### 1. 不同初始化方法
```python
# Kaiming初始化
nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

# Xavier初始化
nn.init.xavier_uniform_(module.weight)
```

### 2. 不同学习率
```python
# 观察不同学习率下权重范数的变化
for lr in [0.0001, 0.001, 0.01]:
    train_model(..., lr=lr, ...)
```

### 3. Layer Normalization对比
```python
# 替换BatchNorm为LayerNorm
self.ln1 = nn.LayerNorm([64, 32, 32])
```

### 4. 更深的网络
```python
# 增加网络深度，观察梯度消失问题
# 带BN vs 不带BN的差异会更明显
```

## 📚 参考资料

### 论文
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)

### 工具文档
- [PyTorch Hooks Tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)
- [TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

### 相关概念
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)
- [On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.html)

## ❓ 常见问题

### Q1: 权重范数一直增长是否正常？
A: 不正常。健康的训练中，权重范数应该：
- 初期增长（学习有用特征）
- 中期稳定（找到好的解）
- 后期微调（fine-tuning）

持续增长可能表示：过拟合、学习率过大、或缺少正则化

### Q2: 如何判断梯度消失？
A: 观察梯度L2范数：
- 浅层梯度范数 << 深层梯度范数 → 梯度消失
- 梯度范数 < 1e-7 → 严重梯度消失
- 使用BatchNorm可以显著缓解

### Q3: 为什么我的实验结果与预期不符？
A: 可能原因：
- 随机种子不同
- GPU vs CPU计算差异
- PyTorch版本不同
- 数据增强的随机性

建议：运行多次取平均值

### Q4: 如何保存实验结果？
A: TensorBoard数据自动保存在 `runs/` 目录：
```bash
# 压缩并下载
tar -czf experiment_results.tar.gz runs/

# 或使用Colab的文件下载
from google.colab import files
files.download('runs/batchnorm_experiment_v2.tar.gz')
```

## 🤝 贡献

欢迎提出改进建议！可以关注的方向：
- 添加更多归一化方法的对比（LayerNorm, GroupNorm, InstanceNorm）
- 支持更多数据集（ImageNet, MNIST, Fashion-MNIST）
- 添加更深的网络架构（ResNet, VGG）
- 实现自动化的实验报告生成

## 📄 许可证

MIT License

---

**Happy Experimenting! 🎉**

如有问题，请查看TensorBoard界面中的详细可视化结果。
