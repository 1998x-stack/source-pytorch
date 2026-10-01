"""
BatchNorm作用验证实验 - 改进版
包含激活值分布监控 + 权重范数监控
使用TensorBoard可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==================== 1. 数据准备 ====================
def prepare_data(batch_size=128):
    """准备CIFAR-10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    
    return trainloader, testloader


# ==================== 2. 模型定义 ====================
class SimpleCNN(nn.Module):
    """不带BatchNorm的CNN"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleCNN_BN(nn.Module):
    """带BatchNorm的CNN"""
    def __init__(self, num_classes=10):
        super(SimpleCNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== 3. 权重范数监控器 ====================
class WeightNormMonitor:
    """监控模型权重的范数"""
    
    def __init__(self, writer, model_name):
        self.writer = writer
        self.model_name = model_name
        self.weight_layers = {}  # 存储需要监控的层
        
    def register_layers(self, model, layer_names):
        """注册需要监控的层"""
        self.weight_layers = {}
        for name, module in model.named_modules():
            if name in layer_names:
                if hasattr(module, 'weight') and module.weight is not None:
                    self.weight_layers[name] = module
                    print(f"  [{self.model_name}] 注册权重监控: {name}")
    
    def compute_weight_norms(self, layer_name, weight_tensor):
        """计算权重的各种范数"""
        weight_np = weight_tensor.detach().cpu().numpy()
        
        # L2范数（Frobenius范数）
        l2_norm = np.linalg.norm(weight_np)
        
        # L1范数
        l1_norm = np.linalg.norm(weight_np.flatten(), ord=1)
        
        # 最大范数（无穷范数）
        max_norm = np.max(np.abs(weight_np))
        
        # 权重的统计量
        mean = np.mean(weight_np)
        std = np.std(weight_np)
        
        # 谱范数（最大奇异值）- 仅对2D矩阵计算
        if len(weight_np.shape) == 2:
            try:
                spectral_norm = np.linalg.norm(weight_np, ord=2)
            except:
                spectral_norm = 0.0
        else:
            spectral_norm = 0.0
        
        return {
            'l2_norm': l2_norm,
            'l1_norm': l1_norm,
            'max_norm': max_norm,
            'spectral_norm': spectral_norm,
            'mean': mean,
            'std': std
        }
    
    def log_weight_norms(self, epoch):
        """记录当前epoch的权重范数到TensorBoard"""
        for layer_name, module in self.weight_layers.items():
            if hasattr(module, 'weight') and module.weight is not None:
                norms = self.compute_weight_norms(layer_name, module.weight)
                
                # 记录到TensorBoard
                for norm_name, norm_value in norms.items():
                    tag = f"{self.model_name}/WeightNorms/{layer_name}/{norm_name}"
                    self.writer.add_scalar(tag, norm_value, epoch)
                
                # 记录权重分布的直方图
                self.writer.add_histogram(
                    f"{self.model_name}/Weights/{layer_name}",
                    module.weight.detach().cpu(),
                    epoch
                )
                
                # 如果有偏置项，也记录
                if hasattr(module, 'bias') and module.bias is not None:
                    self.writer.add_histogram(
                        f"{self.model_name}/Bias/{layer_name}",
                        module.bias.detach().cpu(),
                        epoch
                    )


# ==================== 4. 梯度范数监控器 ====================
class GradientNormMonitor:
    """监控梯度的范数"""
    
    def __init__(self, writer, model_name):
        self.writer = writer
        self.model_name = model_name
        self.gradient_layers = {}
        
    def register_layers(self, model, layer_names):
        """注册需要监控梯度的层"""
        self.gradient_layers = {}
        for name, module in model.named_modules():
            if name in layer_names:
                if hasattr(module, 'weight') and module.weight is not None:
                    self.gradient_layers[name] = module
                    print(f"  [{self.model_name}] 注册梯度监控: {name}")
    
    def log_gradient_norms(self, epoch):
        """记录梯度范数"""
        for layer_name, module in self.gradient_layers.items():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.grad is not None:
                    grad = module.weight.grad.detach().cpu().numpy()
                    
                    # 计算梯度范数
                    grad_l2 = np.linalg.norm(grad)
                    grad_l1 = np.linalg.norm(grad.flatten(), ord=1)
                    grad_max = np.max(np.abs(grad))
                    grad_mean = np.mean(grad)
                    grad_std = np.std(grad)
                    
                    # 记录到TensorBoard
                    self.writer.add_scalar(
                        f"{self.model_name}/GradNorms/{layer_name}/l2_norm",
                        grad_l2, epoch
                    )
                    self.writer.add_scalar(
                        f"{self.model_name}/GradNorms/{layer_name}/max_norm",
                        grad_max, epoch
                    )
                    self.writer.add_scalar(
                        f"{self.model_name}/GradNorms/{layer_name}/mean",
                        grad_mean, epoch
                    )
                    
                    # 记录梯度直方图
                    self.writer.add_histogram(
                        f"{self.model_name}/Gradients/{layer_name}",
                        module.weight.grad.detach().cpu(),
                        epoch
                    )


# ==================== 5. 激活值分布监控器 ====================
class ActivationMonitor:
    """监控激活值的分布统计量"""
    
    def __init__(self, writer, model_name):
        self.writer = writer
        self.model_name = model_name
        self.hooks = []
        self.epoch_stats = {}
        
    def compute_stats(self, tensor):
        """计算张量的统计量"""
        tensor_np = tensor.detach().cpu().numpy().flatten()
        
        if len(tensor_np) == 0:
            return None
            
        mean = np.mean(tensor_np)
        std = np.std(tensor_np)
        variance = std ** 2
        skewness = stats.skew(tensor_np)
        kurtosis = stats.kurtosis(tensor_np)
        median = np.median(tensor_np)
        min_val = np.min(tensor_np)
        max_val = np.max(tensor_np)
        range_val = max_val - min_val
        q25 = np.percentile(tensor_np, 25)
        q75 = np.percentile(tensor_np, 75)
        iqr = q75 - q25
        
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'median': median,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'q25': q25,
            'q75': q75,
            'iqr': iqr
        }
    
    def create_hook(self, layer_name):
        """创建钩子函数"""
        def hook(module, input, output):
            if input is not None and len(input) > 0:
                input_tensor = input[0]
                stats = self.compute_stats(input_tensor)
                
                if stats is not None:
                    if layer_name not in self.epoch_stats:
                        self.epoch_stats[layer_name] = []
                    self.epoch_stats[layer_name].append(stats)
        
        return hook
    
    def register_hooks(self, model, layer_names):
        """为指定层注册钩子"""
        self.remove_hooks()
        
        for name, module in model.named_modules():
            if name in layer_names:
                hook = self.create_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append((name, handle))
    
    def remove_hooks(self):
        """移除所有钩子"""
        for name, handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def log_epoch_stats(self, epoch):
        """将本epoch的统计量记录到TensorBoard"""
        for layer_name, stats_list in self.epoch_stats.items():
            if len(stats_list) == 0:
                continue
                
            # 计算平均统计量
            avg_stats = {}
            for key in stats_list[0].keys():
                avg_stats[key] = np.mean([s[key] for s in stats_list])
            
            # 记录到TensorBoard
            for stat_name, stat_value in avg_stats.items():
                tag = f"{self.model_name}/Activations/{layer_name}/{stat_name}"
                self.writer.add_scalar(tag, stat_value, epoch)
        
        # 清空本epoch的统计
        self.epoch_stats = {}


# ==================== 6. 训练函数 ====================
def train_epoch(model, trainloader, criterion, optimizer, activation_monitor, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1} [训练]', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def test_epoch(model, testloader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_model(model, trainloader, testloader, 
                activation_monitor, weight_monitor, gradient_monitor,
                activation_layers, weight_layers,
                epochs=15, lr=0.001, model_name="model"):
    """完整训练流程"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 注册权重和梯度监控
    weight_monitor.register_layers(model, weight_layers)
    gradient_monitor.register_layers(model, weight_layers)
    
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # 注册激活值监控的钩子
        activation_monitor.register_hooks(model, activation_layers)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, activation_monitor, epoch
        )
        
        # 记录激活值统计
        activation_monitor.log_epoch_stats(epoch)
        activation_monitor.remove_hooks()
        
        # 测试
        test_loss, test_acc = test_epoch(model, testloader, criterion)
        
        # 记录权重范数
        weight_monitor.log_weight_norms(epoch)
        
        # 记录梯度范数
        gradient_monitor.log_gradient_norms(epoch)
        
        # 记录训练指标到TensorBoard
        activation_monitor.writer.add_scalar(f'{model_name}/Loss/train', train_loss, epoch)
        activation_monitor.writer.add_scalar(f'{model_name}/Loss/test', test_loss, epoch)
        activation_monitor.writer.add_scalar(f'{model_name}/Accuracy/train', train_acc, epoch)
        activation_monitor.writer.add_scalar(f'{model_name}/Accuracy/test', test_acc, epoch)
        activation_monitor.writer.add_scalar(f'{model_name}/Learning_Rate', 
                                            optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        
        scheduler.step()
    
    return model


# ==================== 7. 主实验流程 ====================
def main():
    """主实验"""
    # 准备数据
    print("准备数据...")
    trainloader, testloader = prepare_data(batch_size=128)
    
    # 创建TensorBoard writers
    writer_bn = SummaryWriter('runs/batchnorm_experiment_v2/with_bn')
    writer_no_bn = SummaryWriter('runs/batchnorm_experiment_v2/without_bn')
    
    # 创建模型
    print("\n创建模型...")
    model_with_bn = SimpleCNN_BN()
    model_without_bn = SimpleCNN()
    
    # 定义要监控的层
    activation_layers_bn = ['bn1', 'bn2', 'bn3', 'bn_fc']
    activation_layers_no_bn = ['conv1', 'conv2', 'conv3', 'fc1']
    
    weight_layers_bn = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
    weight_layers_no_bn = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
    
    # 创建监控器 - 带BN模型
    print("\n初始化监控器...")
    activation_monitor_bn = ActivationMonitor(writer_bn, "WithBN")
    weight_monitor_bn = WeightNormMonitor(writer_bn, "WithBN")
    gradient_monitor_bn = GradientNormMonitor(writer_bn, "WithBN")
    
    # 创建监控器 - 不带BN模型
    activation_monitor_no_bn = ActivationMonitor(writer_no_bn, "WithoutBN")
    weight_monitor_no_bn = WeightNormMonitor(writer_no_bn, "WithoutBN")
    gradient_monitor_no_bn = GradientNormMonitor(writer_no_bn, "WithoutBN")
    
    # 训练带BN的模型
    model_with_bn = train_model(
        model_with_bn, trainloader, testloader,
        activation_monitor_bn, weight_monitor_bn, gradient_monitor_bn,
        activation_layers_bn, weight_layers_bn,
        epochs=15, lr=0.001, model_name="带BatchNorm"
    )
    
    # 训练不带BN的模型
    model_without_bn = train_model(
        model_without_bn, trainloader, testloader,
        activation_monitor_no_bn, weight_monitor_no_bn, gradient_monitor_no_bn,
        activation_layers_no_bn, weight_layers_no_bn,
        epochs=15, lr=0.001, model_name="不带BatchNorm"
    )
    
    # 关闭writers
    writer_bn.close()
    writer_no_bn.close()
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    print("\n运行以下命令启动TensorBoard:")
    print("tensorboard --logdir=runs/batchnorm_experiment_v2")
    print("\n然后在浏览器中打开: http://localhost:6006")
    print("\n在TensorBoard中可以查看:")
    print("  📊 SCALARS tab:")
    print("     - 训练/测试损失和准确率")
    print("     - 激活值分布统计 (Activations)")
    print("     - 权重范数 (WeightNorms): L1, L2, Max, Spectral")
    print("     - 梯度范数 (GradNorms)")
    print("  📈 HISTOGRAMS tab:")
    print("     - 权重分布演变 (Weights)")
    print("     - 梯度分布演变 (Gradients)")
    print("     - 偏置分布演变 (Bias)")
    print("\n提示: 在SCALARS中选择多个指标进行对比")


if __name__ == "__main__":
    main()
