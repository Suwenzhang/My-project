import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pytorch_msssim import SSIM
import numpy as np
from model import resnet18, nt_xent_loss  # 引入优化后的 ResNet 和对比学习损失函数
import os
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler

# 设置更大的max_split_size_mb，减少显存碎片化
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

    def train(self, train_loader, val_loader=None, epochs=50):
        self.model.train()
        scaler = GradScaler()  # 正确实例化 torch.cuda.amp.GradScaler
        accumulation_steps = 8  # 增加梯度累积步数为8

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

                with autocast():
                    classification_output, projection_output = self.model(noisy_images)
                    contrastive_loss = nt_xent_loss(projection_output, temperature=self.temperature)
                    loss = self.compute_loss(classification_output, labels, epoch, epochs)
                    total_loss_step = loss + contrastive_loss
                    total_loss_step = total_loss_step / accumulation_steps  # 梯度累积

                # 确保传递的 total_loss_step 是一个张量
                scaler.scale(total_loss_step).backward()  # 正确传递 total_loss_step

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
                    self.save_model('resnet_pretrained_optimized.pth')
                else:
                    self.early_stop_counter += 1

                print(f"Validation Loss: {val_loss:.4f}, Early Stop Counter: {self.early_stop_counter}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Training stopped.")
                    break

            if self.scheduler:
                self.scheduler.step(val_loss)

        print("Pretraining completed!")

    def apply_mixup(self, images, labels, alpha=0.2):
        """
        针对少数类标签的数据进行 Mixup 数据增强
        :param images: 输入图像
        :param labels: 输入标签
        :param alpha: Mixup 参数
        """
        unique_labels, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        threshold = 0.2 * total_samples
        minority_labels = unique_labels[counts < threshold]

        minority_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in minority_labels:
            minority_mask |= labels.eq(label)  # 等价于 isin 功能

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

        unique_labels, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        threshold = 0.2 * total_samples
        minority_labels = unique_labels[counts < threshold]

        weights = torch.ones_like(labels, dtype=torch.float32)

        for minority_label in minority_labels:
            weights[labels == minority_label] = 1.5

        loss_ce = criterion_ce(classification_output, labels)

        dynamic_weight = epoch / num_epochs
        base_loss = weights * (0.7 * loss_ce)

        return base_loss.mean()

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


def get_cifar10_dataloader(batch_size=4, train=True):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    return dataloader


def main():
    model = resnet18(num_classes=10)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

    train_loader = get_cifar10_dataloader(batch_size=4, train=True)
    val_loader = get_cifar10_dataloader(batch_size=4, train=False)

    trainer = ResNetTrainer(model, optimizer, scheduler, patience=20, lambda_pos=0.002, lambda_neg=0.008,
                            memory_size=32)
    trainer.train(train_loader, val_loader=val_loader, epochs=50)
    trainer.save_model('resnet_pretrained_optimized.pth')


if __name__ == "__main__":
    main()
