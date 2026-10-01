# BatchNorm作用验证实验

本实验验证BatchNorm对训练稳定性的影响，通过hook机制获取每层的分布统计量（均值、方差、偏度、峰度等），并使用TensorBoard进行可视化。

## 实验目的

验证BatchNorm的核心作用：
1. **分布规范化**：使层输入分布接近标准正态分布（均值≈0，方差≈1）
2. **训练稳定性**：减少内部协变量偏移，使分布统计量更稳定
3. **收敛加速**：加快模型收敛速度
4. **性能提升**：提高模型的最终性能

## 文件说明

- `batchnorm_experiment.py`: 完整的Python脚本版本
- `batchnorm_experiment.ipynb`: Colab Notebook版本（推荐）

## 在Google Colab中运行

### 方法1：直接上传Notebook

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 选择 "文件" -> "上传笔记本"
3. 上传 `batchnorm_experiment.ipynb`
4. 按顺序执行所有单元格

### 方法2：从GitHub导入

1. 将文件上传到GitHub仓库
2. 在Colab中选择 "文件" -> "从GitHub打开"
3. 输入仓库URL

## 实验流程

### 1. 数据准备
使用CIFAR-10数据集，包含60,000张32×32的彩色图像

### 2. 模型对比
- **带BatchNorm的模型**：在每个卷积层和全连接层后添加BatchNorm
- **不带BatchNorm的模型**：相同架构但没有BatchNorm层

### 3. 统计量监控
使用Hook机制捕获每层的输入，计算以下统计量：
- 均值（mean）
- 标准差（std）
- 方差（variance）
- 偏度（skewness）
- 峰度（kurtosis）
- 中位数（median）
- 最小值/最大值（min/max）
- 范围（range）
- 四分位距（IQR）

### 4. TensorBoard可视化
实时记录和可视化：
- 训练/测试损失和准确率
- 每层的分布统计量随epoch的变化
- 两个模型的对比

## 使用说明

### 在Colab中

```python
# 1. 安装依赖（第一个cell）
%load_ext tensorboard
!pip install torch torchvision scipy -q

# 2. 启动TensorBoard（在训练前运行）
%tensorboard --logdir runs/batchnorm_experiment

# 3. 运行训练（按顺序执行所有cell）
# 训练过程会自动记录到TensorBoard

# 4. 在TensorBoard界面查看结果
```

### 本地运行Python脚本

```bash
# 安装依赖
pip install torch torchvision scipy tensorboard

# 运行实验
python batchnorm_experiment.py

# 启动TensorBoard
tensorboard --logdir=runs/batchnorm_experiment

# 在浏览器打开
# http://localhost:6006
```

## TensorBoard使用指南

### 查看标量指标（SCALARS）

1. **性能指标**
   - `带BatchNorm/Loss/train` vs `不带BatchNorm/Loss/train`
   - `带BatchNorm/Accuracy/test` vs `不带BatchNorm/Accuracy/test`

2. **分布统计量**
   - `WithBN/bn1/mean` - 第一个BN层的均值
   - `WithBN/bn1/std` - 第一个BN层的标准差
   - `WithBN/bn1/skewness` - 偏度
   - `WithBN/bn1/kurtosis` - 峰度
   - 对比不同层和不同模型

### 查看直方图（HISTOGRAMS）

- 查看激活值的分布变化
- 比较不同epoch的分布

### TensorBoard技巧

1. **多指标对比**：在左侧选择多个实验，右下角勾选要对比的指标
2. **平滑曲线**：调整左侧的"Smoothing"滑块
3. **缩放**：使用鼠标滚轮或拖动坐标轴
4. **下载数据**：点击左下角的下载按钮

## 预期实验结果

### 带BatchNorm的模型

- ✅ 均值接近0
- ✅ 标准差接近1
- ✅ 偏度接近0（分布对称）
- ✅ 峰度接近0（接近正态分布）
- ✅ 统计量在训练过程中保持稳定
- ✅ 收敛速度更快
- ✅ 最终测试准确率更高

### 不带BatchNorm的模型

- ❌ 均值和标准差可能偏离0和1
- ❌ 偏度和峰度较大（分布不规则）
- ❌ 统计量在训练过程中波动较大
- ❌ 收敛速度较慢
- ❌ 可能出现梯度消失/爆炸

## 实验参数

- **数据集**：CIFAR-10
- **Batch Size**：128
- **训练轮数**：15 epochs
- **学习率**：0.001（Adam优化器）
- **学习率调度**：每5个epoch衰减0.5倍
- **Dropout**：0.3

## 监控的层

### 带BN模型
- `bn1`: 第一个BatchNorm层
- `bn2`: 第二个BatchNorm层
- `bn3`: 第三个BatchNorm层
- `bn_fc`: 全连接层的BatchNorm

### 不带BN模型
- `conv1`: 第一个卷积层
- `conv2`: 第二个卷积层
- `conv3`: 第三个卷积层
- `fc1`: 第一个全连接层

## 关键观察点

### 1. 分布稳定性
观察统计量随epoch的变化：
- 带BN：应该快速稳定并保持稳定
- 不带BN：可能持续波动

### 2. 分布特征
观察是否接近标准正态分布：
- 均值 ≈ 0
- 标准差 ≈ 1
- 偏度 ≈ 0
- 峰度 ≈ 0

### 3. 训练动态
- 损失下降速度
- 准确率提升速度
- 训练曲线的平滑程度

## 常见问题

### Q1: TensorBoard显示不出来？
A: 确保运行了 `%tensorboard --logdir runs/batchnorm_experiment` 这个cell，并且在训练开始前运行。

### Q2: 训练太慢？
A: 在Colab中，确保使用GPU：Runtime -> Change runtime type -> GPU

### Q3: 内存不足？
A: 减小batch_size，例如从128改为64

### Q4: 想修改训练参数？
A: 在调用 `train_model()` 函数时修改参数：
```python
train_model(..., epochs=10, lr=0.0005, ...)
```

## 扩展实验

可以尝试的变化：
1. 使用不同的数据集（MNIST、Fashion-MNIST）
2. 调整网络深度
3. 比较不同的归一化方法（LayerNorm、GroupNorm）
4. 添加更多的统计量监控
5. 可视化权重的分布变化

## 参考资料

- [Batch Normalization论文](https://arxiv.org/abs/1502.03167)
- [TensorBoard文档](https://www.tensorflow.org/tensorboard)
- [PyTorch Hook教程](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)

## 许可证

MIT License
