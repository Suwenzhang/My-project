import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pytorch_msssim import SSIM
import numpy as np
from model import resnet18, nt_xent_loss  # 引入优化后的 ResNet 和对比学习损失函数
from torch.cuda.amp import autocast, GradScaler
import os

# 设置更大的 max_split_size_mb，减少显存碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'


class ResNetTrainer:
    def __init__(self, model, optimizer, scheduler=None, patience=20,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lambda_pos=0.002, lambda_neg=0.008, memory_size=32, temperature=0.5):
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
        self.previous_contrastive_loss = float('inf')
        self.previous_classification_loss = float('inf')
        self.beta = 0.9  # 动量系数，用于平滑损失
        self.smoothed_contrastive_loss = 0
        self.smoothed_classification_loss = 0

    def train(self, train_loader, val_loader=None, epochs=50):
        self.model.train()
        scaler = GradScaler()  # 使用混合精度的梯度缩放
        accumulation_steps = 8  # 梯度累积步数

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            self.optimizer.zero_grad()

            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device).float()
                labels = labels.to(self.device)

                # 动态选择是否进行Mixup
                images, labels = self.apply_mixup(images, labels)

                # 蒸馏数据的动态噪声注入
                noisy_images = self.dynamic_noise_injection(images, labels, epoch, epochs)

                with autocast():  # 使用混合精度计算
                    classification_output, projection_output = self.model(noisy_images)
                    contrastive_loss = nt_xent_loss(projection_output, temperature=self.temperature)
                    classification_loss = self.compute_loss(classification_output, labels, epoch, epochs)

                    # 动态平衡对比学习和分类任务损失
                    alpha_contrastive, alpha_classification = self.get_dynamic_alpha(contrastive_loss, classification_loss, epoch, epochs)
                    total_loss_step = alpha_contrastive * contrastive_loss + alpha_classification * classification_loss
                    total_loss_step = total_loss_step / accumulation_steps

                scaler.scale(total_loss_step).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                total_loss += total_loss_step.item() * accumulation_steps
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

            if val_loader:
                val_loss = self.validate(val_loader, epoch, epochs)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stop_counter = 0
                    self.save_model('resnet_pretrained_optimized4.pth')
                else:
                    self.early_stop_counter += 1

                print(f"Validation Loss: {val_loss:.4f}, Early Stop Counter: {self.early_stop_counter}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Training stopped.")
                    break

            if self.scheduler:
                self.scheduler.step(val_loss)

        print("Pretraining completed!")

    def get_dynamic_alpha(self, contrastive_loss, classification_loss, epoch, epochs):
        """
        根据损失变化率和训练进度，动态调整对比学习和分类任务的损失权重。
        """
        # 平滑更新损失
        self.smoothed_contrastive_loss = self.beta * self.smoothed_contrastive_loss + (
                    1 - self.beta) * contrastive_loss.item()
        self.smoothed_classification_loss = self.beta * self.smoothed_classification_loss + (
                    1 - self.beta) * classification_loss.item()

        # 计算损失变化率
        contrastive_loss_change = abs(self.smoothed_contrastive_loss - contrastive_loss.item()) / max(
            self.smoothed_contrastive_loss, 1e-8)
        classification_loss_change = abs(self.smoothed_classification_loss - classification_loss.item()) / max(
            self.smoothed_classification_loss, 1e-8)

        # 根据训练进度动态调整 alpha
        progress = epoch / epochs
        alpha_dynamic = 0.5 * (1 - progress)  # 进度越晚，越倾向于对比学习
        total_change = contrastive_loss_change + classification_loss_change

        # 动态调整的 alpha 权重
        alpha_contrastive = alpha_dynamic + (contrastive_loss_change / total_change) * (1 - alpha_dynamic)
        alpha_classification = (1 - alpha_dynamic) + (classification_loss_change / total_change) * alpha_dynamic

        return alpha_contrastive, alpha_classification

    def apply_mixup(self, images, labels, alpha=0.2):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        threshold = 0.2 * total_samples
        minority_labels = unique_labels[counts < threshold]

        minority_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in minority_labels:
            minority_mask |= labels.eq(label)

        if minority_mask.sum() == 0:
            return images, labels

        lam = np.random.beta(alpha, alpha)
        indices = torch.randperm(images.size(0))

        images_mixup = lam * images + (1 - lam) * images[indices]
        labels_mixup = lam * labels + (1 - lam) * labels[indices]

        return images_mixup, labels_mixup

    def dynamic_noise_injection(self, images, labels, epoch, num_epochs):
        dynamic_factor = 0.5 * (1 - epoch / num_epochs)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)

        threshold = 0.2 * total_samples
        minority_labels = unique_labels[counts < threshold]
        majority_labels = unique_labels[counts >= threshold]

        noise_factor = torch.zeros_like(labels, dtype=torch.float32)

        for minority_label in minority_labels:
            noise_factor[labels == minority_label] = 0.5 * dynamic_factor
        for majority_label in majority_labels:
            noise_factor[labels == majority_label] = 0.2 * dynamic_factor

        noisy_images = images + noise_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3) * torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        return noisy_images

    def compute_loss(self, classification_output, labels, epoch, num_epochs):
        criterion_ce = nn.CrossEntropyLoss()

        # 将标签转换为 Long 类型
        labels = labels.long()

        return criterion_ce(classification_output, labels)

    def validate(self, val_loader, epoch, epochs):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device).float()
                labels = labels.to(self.device)

                noisy_images = self.dynamic_noise_injection(images, labels, epoch, epochs)
                classification_output, _ = self.model(noisy_images)
                loss = self.compute_loss(classification_output, labels, epoch, epochs)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.model.train()
        return avg_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def get_dataloader(dataset_name, batch_size=4, train=True):
    if dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)


def main():
    # 修改为 100 类
    model = resnet18(num_classes=100)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

    # 使用 CIFAR-100 作为示例数据集
    train_loader = get_dataloader('cifar100', batch_size=64, train=True)
    val_loader = get_dataloader('cifar100', batch_size=64, train=False)

    trainer = ResNetTrainer(model, optimizer, scheduler, patience=20)
    trainer.train(train_loader, val_loader=val_loader, epochs=100)
    trainer.save_model('resnet_pretrained_optimized_cifar100.pth')


if __name__ == "__main__":
    main()
