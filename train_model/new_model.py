import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ["resnet10", "resnet18", "resnet34"]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # Set track_running_stats=True to improve stability during validation
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=True),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetWithProjection(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200, projection_dim=128):
        super(ResNetWithProjection, self).__init__()
        self.in_planes = 64
        self.projection_dim = projection_dim

        # ResNet layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim)  # Adding BatchNorm1d for better stability
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Adaptive average pooling to get feature map size (batch_size, 512, 1, 1)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)  # Flattening the output

        # Linear classification head
        classification_output = self.linear(out)

        # Projection head for contrastive learning
        projection_output = self.projection_head(out)

        return classification_output, projection_output


def resnet10(num_classes=200, projection_dim=128):
    return ResNetWithProjection(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, projection_dim=projection_dim)


def resnet18(num_classes=200, projection_dim=128):
    return ResNetWithProjection(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, projection_dim=projection_dim)


def resnet34(num_classes=200, projection_dim=128):
    return ResNetWithProjection(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, projection_dim=projection_dim)


# NT-Xent Loss for Contrastive Learning
def nt_xent_loss(projections, temperature=0.5):
    """
    Compute the NT-Xent loss for contrastive learning.
    projections: tensor of shape (batch_size, projection_dim)
    temperature: temperature hyperparameter for the loss.
    """
    batch_size = projections.shape[0]

    # Normalize the projections to the unit sphere
    projections = F.normalize(projections, dim=1)

    # Compute similarity matrix, subtract the max for numerical stability
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    similarity_matrix -= torch.max(similarity_matrix, dim=1, keepdim=True)[0]

    # Labels for positive pairs
    labels = torch.arange(batch_size).cuda()

    # Compute NT-Xent loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


# Optimizer, Learning rate scheduler, and Gradient Clipping
def get_optimizer_and_scheduler(model, learning_rate=1e-4, weight_decay=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Using ReduceLROnPlateau to adjust learning rate based on validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    return optimizer, scheduler


# Function to clip gradients
def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
