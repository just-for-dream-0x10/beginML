# Image Processing with PyTorch

## 1. Basic Image Operations

### Loading and Displaying Images
```python
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load an image
image = Image.open('path/to/image.jpg')

# Convert to tensor
transform = transforms.ToTensor()
tensor_image = transform(image)

# Display image
plt.imshow(tensor_image.permute(1, 2, 0))
plt.title("Original Image")
plt.axis('off')
plt.show()

# Display image channels separately
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
channels = ['Red', 'Green', 'Blue']
for i, ax in enumerate(axes):
    ax.imshow(tensor_image[i].numpy(), cmap='gray')
    ax.set_title(f'{channels[i]} Channel')
    ax.axis('off')
plt.show()
```

### Image Transformations

#### Basic Transformations
```python
# Common transformations
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Crop central 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Apply transformations
transformed_image = transform(image)

# Display original vs transformed
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(transformed_image.permute(1, 2, 0))
axes[1].set_title('Transformed')
axes[1].axis('off')
plt.show()
```

#### Advanced Transformations
```python
# Create a visualization function
from torchvision.utils import make_grid

def show_transforms(image, transforms_list):
    """Show original image and all transformations"""
    transformed_images = []
    for transform in transforms_list:
        transformed = transform(image)
        transformed_images.append(transformed)
    
    grid = make_grid(transformed_images, nrow=3)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Transformations Comparison")
    plt.axis('off')
    plt.show()

# Define different transformations
transforms_list = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # Horizontal flip
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomRotation(45),  # Rotate 45 degrees
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Translation
        transforms.ToTensor()
    ])
]

# Visualize transformations
show_transforms(image, transforms_list)
```

## 2. Image Preprocessing

### Color Space Conversion

#### Grayscale Conversion
```python
# Convert to grayscale with visualization
gray_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Convert and display
gray_image = gray_transform(image)
plt.figure(figsize=(8, 8))
plt.imshow(gray_image.squeeze(), cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()
```

#### HSV Conversion
```python
# Convert RGB to HSV color space
def rgb_to_hsv(rgb_image):
    """Convert RGB to HSV color space"""
    r, g, b = rgb_image.chunk(3, dim=0)
    
    maxc = torch.max(rgb_image, dim=0)[0]
    minc = torch.min(rgb_image, dim=0)[0]
    
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc == 0, torch.zeros_like(maxc), delta / maxc)
    
    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta
    
    h = torch.zeros_like(maxc)
    h[maxc == r] = bc[maxc == r] - gc[maxc == r]
    h[maxc == g] = 2.0 + rc[maxc == g] - bc[maxc == g]
    h[maxc == b] = 4.0 + gc[maxc == b] - rc[maxc == b]
    
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v])

# Convert and display HSV channels
hsv_image = rgb_to_hsv(tensor_image)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
channels = ['Hue', 'Saturation', 'Value']
for i, ax in enumerate(axes):
    ax.imshow(hsv_image[i], cmap='gray')
    ax.set_title(channels[i])
    ax.axis('off')
plt.show()
```

### Image Normalization

#### Custom Normalization
```python
# Custom normalization with visualization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean, std=std)
normalized_image = normalize(tensor_image)

# Display original vs normalized
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(tensor_image.permute(1, 2, 0))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(normalized_image.permute(1, 2, 0))
axes[1].set_title('Normalized')
axes[1].axis('off')
plt.show()

# Show pixel distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].hist(tensor_image.numpy().ravel(), bins=256, color='blue', alpha=0.7)
axes[0].set_title('Original Pixel Distribution')
axes[1].hist(normalized_image.numpy().ravel(), bins=256, color='red', alpha=0.7)
axes[1].set_title('Normalized Pixel Distribution')
plt.show()
```

## 3. Advanced Image Processing

### Color Space Operations

#### HSV Color Space
```python
# Convert RGB to HSV color space
def rgb_to_hsv(rgb_image):
    """Convert RGB to HSV color space"""
    r, g, b = rgb_image.chunk(3, dim=0)
    
    maxc = torch.max(rgb_image, dim=0)[0]
    minc = torch.min(rgb_image, dim=0)[0]
    
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc == 0, torch.zeros_like(maxc), delta / maxc)
    
    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta
    
    h = torch.zeros_like(maxc)
    h[maxc == r] = bc[maxc == r] - gc[maxc == r]
    h[maxc == g] = 2.0 + rc[maxc == g] - bc[maxc == g]
    h[maxc == b] = 4.0 + gc[maxc == b] - rc[maxc == b]
    
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v])

# Visualize HSV conversion
hsv_image = rgb_to_hsv(tensor_image)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(tensor_image.permute(1, 2, 0))
axes[0].set_title('Original')
axes[0].axis('off')

channels = ['Hue', 'Saturation', 'Value']
for i, ax in enumerate(axes[1:], 1):
    ax.imshow(hsv_image[i-1], cmap='gray')
    ax.set_title(channels[i-1])
    ax.axis('off')
plt.show()
```

### Image Enhancement

#### Gamma Correction
```python
# Gamma correction with visualization
def gamma_correction(image, gamma=1.0):
    return image.pow(gamma)

# Create different gamma corrections
gammas = [0.5, 1.0, 2.0]
fig, axes = plt.subplots(1, len(gammas) + 1, figsize=(15, 5))
axes[0].imshow(tensor_image.permute(1, 2, 0))
axes[0].set_title('Original')
axes[0].axis('off')

for i, gamma in enumerate(gammas, 1):
    corrected = gamma_correction(tensor_image, gamma)
    axes[i].imshow(corrected.permute(1, 2, 0))
    axes[i].set_title(f'Gamma={gamma}')
    axes[i].axis('off')
plt.show()
```

#### Histogram Equalization
```python
# Histogram equalization with visualization
def histogram_equalization(image):
    """Apply histogram equalization to an image"""
    image = image * 255  # Convert to 0-255 range
    image = image.to(torch.uint8)
    
    # Calculate histogram
    hist = torch.histc(image.float(), bins=256, min=0, max=255)
    
    # Calculate cumulative distribution
    cdf = hist.cumsum(dim=0)
    cdf = cdf / cdf[-1] * 255
    
    # Apply equalization
    equalized = torch.zeros_like(image)
    for i in range(256):
        equalized[image == i] = cdf[i]
    
    return equalized.float() / 255  # Convert back to 0-1 range

# Visualize equalization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tensor_image.permute(1, 2, 0))
axes[0].set_title('Original')
axes[0].axis('off')

equalized = histogram_equalization(tensor_image)
axes[1].imshow(equalized.permute(1, 2, 0))
axes[1].set_title('Equalized')
axes[1].axis('off')

# Show histograms
axes[2].hist(tensor_image.numpy().ravel(), bins=256, alpha=0.5, label='Original')
axes[2].hist(equalized.numpy().ravel(), bins=256, alpha=0.5, label='Equalized')
axes[2].legend()
plt.show()
```

## 4. Image Augmentation

### Basic Augmentations with Visualization
```python
# Create a visualization function
def visualize_augmentations(image, transforms_list, title='Augmentations'):
    """Visualize multiple augmentations"""
    transformed_images = []
    for transform in transforms_list:
        transformed = transform(image)
        transformed_images.append(transformed)
    
    grid = make_grid(transformed_images, nrow=3)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Basic augmentations
transforms_list = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # Horizontal flip
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomRotation(45),  # Rotate 45 degrees
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Translation
        transforms.ToTensor()
    ])
]

# Visualize basic augmentations
visualize_augmentations(image, transforms_list)
```

### Advanced Augmentations with Albumentations
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Advanced augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(p=0.2),
    ToTensorV2()
])

# Visualize advanced augmentations
advanced_transforms = [
    A.Compose([A.HorizontalFlip(p=1), ToTensorV2()]),
    A.Compose([A.GaussNoise(p=1), ToTensorV2()]),
    A.Compose([A.CoarseDropout(p=1), ToTensorV2()])
]

visualize_augmentations(image, advanced_transforms, 'Advanced Augmentations')
```

## 5. Best Practices

### Memory Management
```python
# Efficient image loading
def load_image(path):
    with torch.no_grad():
        image = Image.open(path)
        return transform(image)

# Use DataLoader with appropriate batch size
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)
```

### Performance Optimization
```python
# Use compiled transforms
@torch.compile
def process_image(image):
    return transform(image)

# Use mixed precision for augmentations
with torch.cuda.amp.autocast():
    augmented_image = transform(image)
```

## 6. Common Pitfalls

### Memory Issues
- Avoid loading large images into memory all at once
- Use appropriate batch sizes based on available GPU memory
- Release unused tensors with `del` and call `torch.cuda.empty_cache()`

### Data Augmentation
- Be careful not to over-augment
- Some augmentations might not be suitable for certain tasks
- Always validate if augmentations make sense for your specific task

### Normalization
- Always use the correct mean and standard deviation for your dataset
- Be consistent with normalization across training and inference
- Consider using dataset-specific statistics when possible
