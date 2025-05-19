# Convolutional Neural Networks (CNNs) in PyTorch

## 1. Basic CNN Architecture

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## 2. Advanced CNN Architectures

### ResNet Block
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
```

### EfficientNet Block
```python
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, stride):
        super().__init__()
        expanded = in_channels * expansion
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(),
            
            nn.Conv2d(expanded, expanded, kernel_size, stride, padding=kernel_size//2,
                      groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(),
            
            nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)
```

## 3. Transfer Learning

```python
from torchvision import models

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained model
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
```

## 4. Optimization Techniques

### Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# In training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)
```

### Mixed Precision Training
```python
from torch.cuda.amp import GradScaler, autocast

# Initialize scaler
scaler = GradScaler()

# Training loop
for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 5. Best Practices

### Model Architecture Design
- Start with smaller models and gradually increase complexity
- Use appropriate kernel sizes for your task (3x3 is common)
- Consider using depthwise separable convolutions for efficiency
- Use batch normalization after convolutions (except after ReLU)

### Training Tips
- Always use data augmentation during training
- Use appropriate learning rate schedules
- Monitor validation loss and accuracy
- Consider using early stopping

### Performance Optimization
- Use batch normalization for better convergence
- Implement mixed precision training
- Use appropriate activation functions (ReLU, SiLU, etc.)
- Consider using model parallelism for large models

## 6. Common Pitfalls

### Overfitting
- Too many layers for small datasets
- Not enough data augmentation
- Lack of regularization

### Underfitting
- Too simple model architecture
- Insufficient training time
- Learning rate too low

### Memory Issues
- Large batch sizes without sufficient GPU memory
- Deep architectures without proper memory management
- Lack of gradient checkpointing for large models
