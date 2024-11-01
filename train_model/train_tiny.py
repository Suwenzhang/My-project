import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
from model import resnet18, nt_xent_loss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# 设置显存配置，避免显存碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

class FocalLoss(nn.Module):
    """Focal Loss 实现，用于处理标签不平衡问题"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, labels):
        ce_loss = self.ce_loss(outputs, labels)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

class ResNetTrainer:
    def __init__(self, model, optimizer, scheduler=None, patience=20, device='cuda', lambda_pos=0.002,
                 lambda_neg=0.008, memory_size=32, temperature=0.5, log_dir='logs'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_loss = np.inf
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.memory_size = memory_size
        self.temperature = temperature
        self.early_stop_counter = 0
        self.scaler = GradScaler()  # 混合精度训练
        self.writer = SummaryWriter(log_dir=log_dir)
        self.accumulation_steps = self.get_dynamic_accumulation_steps()

    def get_dynamic_accumulation_steps(self):
        """根据显存大小动态调整梯度累积步数"""
        memory_available = torch.cuda.get_device_properties(0).total_memory
        if memory_available > 12e9:
            return 2  # 从较小的累积步数开始
        return 4

    def train(self, train_loader, val_loader=None, epochs=50, minority_labels=None):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            self.optimizer.zero_grad()

            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()

                images, labels = self.apply_mixup(images, labels)  # 应用 Mixup 数据增强
                noisy_images = self.dynamic_noise_injection(images, labels, epoch, epochs)

                with autocast():  # 混合精度训练
                    classification_output, projection_output = self.model(noisy_images)
                    contrastive_loss = nt_xent_loss(projection_output, temperature=self.temperature)
                    classification_loss = self.compute_loss(classification_output, labels, epoch, epochs)

                    # 动态调整损失权重
                    total_loss_step = self.get_dynamic_alpha(contrastive_loss, classification_loss, epoch, epochs)
                    total_loss_step = total_loss_step / self.accumulation_steps

                self.scaler.scale(total_loss_step).backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)  # 添加梯度裁剪

                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_loss += total_loss_step.item() * self.accumulation_steps
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

            self.writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
            if val_loader:
                val_loss, minority_accuracy = self.validate(val_loader, epoch, epochs, minority_labels)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/minority', minority_accuracy, epoch)

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stop_counter = 0
                    self.save_model(f'checkpoints/epoch_{epoch}_best.pth')
                else:
                    self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            if self.scheduler:
                self.scheduler.step(val_loss)

        self.writer.close()

    def get_dynamic_alpha(self, contrastive_loss, classification_loss, epoch, epochs):
        """动态调整对比学习和分类任务的损失权重"""
        alpha_contrastive = (1 - epoch / (epochs * 1.5)) * 0.3
        alpha_classification = 1 - alpha_contrastive
        return alpha_contrastive * contrastive_loss + alpha_classification * classification_loss

    def apply_mixup(self, images, labels, alpha=0.2):
        """Mixup 数据增强"""
        lam = np.random.beta(alpha, alpha)
        indices = torch.randperm(images.size(0))
        mixed_images = lam * images + (1 - lam) * images[indices]
        return mixed_images, labels

    def dynamic_noise_injection(self, images, labels, epoch, num_epochs):
        """动态噪声注入，增强训练"""
        dynamic_factor = 0.2 * (1 - epoch / num_epochs)
        noise_factor = torch.rand_like(images) * dynamic_factor
        return torch.clamp(images + noise_factor, 0, 1)

    def compute_loss(self, classification_output, labels, epoch, num_epochs):
        """使用 Focal Loss 代替交叉熵损失"""
        criterion = FocalLoss(alpha=1.5, gamma=2)  # 调整 alpha 来加强对少数类的关注
        return criterion(classification_output, labels)

    def validate(self, val_loader, epoch, epochs, minority_labels):
        """验证过程，增加对少数类的评估"""
        self.model.eval()
        total_loss = 0
        minority_correct = 0
        total_minority = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()
                noisy_images = self.dynamic_noise_injection(images, labels, epoch, epochs)

                classification_output, _ = self.model(noisy_images)
                loss = self.compute_loss(classification_output, labels, epoch, epochs)
                total_loss += loss.item()

                preds = classification_output.argmax(dim=1)

                # 使用 torch.eq 代替 torch.isin
                minority_mask = torch.zeros_like(labels, dtype=torch.bool)
                for minority_label in minority_labels:
                    minority_mask |= (labels == minority_label)

                minority_correct += (preds[minority_mask] == labels[minority_mask]).sum().item()
                total_minority += minority_mask.sum().item()

        avg_loss = total_loss / len(val_loader)
        minority_accuracy = minority_correct / total_minority if total_minority > 0 else 0
        self.model.train()
        return avg_loss, minority_accuracy

    def save_model(self, epoch, val_loss, minority_accuracy):
        """保存模型权重，文件名中包含轮次、验证损失和少数类准确率"""
        directory = 'checkpoints'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 格式化保存的模型文件名，包含 epoch、val_loss 和 minority_accuracy
        model_filename = f'{directory}/epoch_{epoch}_val_loss_{val_loss:.4f}_minority_acc_{minority_accuracy:.4f}.pth'

        torch.save(self.model.state_dict(), model_filename)

def get_sampler(dataset):
    """获取加权随机采样器，增强少数类样本的采样概率"""
    targets = [label for _, label in dataset.imgs]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets]

    return WeightedRandomSampler(sample_weights, len(sample_weights))

def get_dataloader(dataset_name, batch_size=64, train=True):
    """加载数据集并应用加权采样"""
    data_dir = 'C:/Users/Zan/Desktop/train_model/tiny-imagenet-200'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if dataset_name == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) if train else transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset_dir = os.path.join(data_dir, 'train' if train else 'val')
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

        if train:
            sampler = get_sampler(dataset)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        raise ValueError("Unknown dataset")

def main():
    model = resnet18(num_classes=200)  # 初始化模型
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)  # AdamW 优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 加载训练和验证数据集
    train_loader = get_dataloader('tiny_imagenet', batch_size=64, train=True)
    val_loader = get_dataloader('tiny_imagenet', batch_size=64, train=False)

    # 定义少数类标签列表 (根据数据集定义)
    minority_labels = torch.tensor([0, 1, 2, 3, 4, 5])  # 假设 0-5 是少数类

    trainer = ResNetTrainer(model, optimizer, scheduler, patience=20)
    trainer.train(train_loader, val_loader=val_loader, epochs=100, minority_labels=minority_labels)
    trainer.save_model('resnet_pretrained_tiny_imagenet.pth')

if __name__ == "__main__":
    main()
