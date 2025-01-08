import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import timm  # For loading Xception, SENet, etc.

# Define LeNet-5 Model
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define AlexNet Model
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Add an adaptive pooling layer after convolutions
            nn.AdaptiveAvgPool2d((6, 6))  # This is the critical change
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # Adjust the input size accordingly
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        
        # Flatten the feature map
        x = torch.flatten(x, 1)  # Flatten to (batch_size, channels * height * width)
        
        x = self.classifier(x)
        return x


# Define GoogLeNet Model
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.model = torchvision.models.googlenet(pretrained=False, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define VGGNet Model
class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=False)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define Xception Model (from timm library)
class Xception(nn.Module):
    def __init__(self, num_classes=10):
        super(Xception, self).__init__()
        self.model = timm.create_model('xception', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Define SENet Model (from timm library)
class SENet(nn.Module):
    def __init__(self, num_classes=10):
        super(SENet, self).__init__()
        self.model = timm.create_model('seresnet18', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Data Transformations and Loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

# Device and Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {
    'LeNet-5': LeNet5(num_classes=10).to(device),
    'AlexNet': AlexNet(num_classes=10).to(device),
    'GoogLeNet': GoogLeNet(num_classes=10).to(device),
    'VGGNet': VGGNet(num_classes=10).to(device),
    'Xception': Xception(num_classes=10).to(device),
    'SENet': SENet(num_classes=10).to(device)
}

criterion = nn.CrossEntropyLoss()
optimizers = {
    name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()
}
schedulers = {
    name: StepLR(optimizer, step_size=10, gamma=0.1) for name, optimizer in optimizers.items()
}

# Training and Evaluation for all models
num_epochs = 20
train_loss_dict = {name: [] for name in models}
test_accuracy_dict = {name: [] for name in models}

for epoch in range(num_epochs):
    for name, model in models.items():
        model.train()
        optimizer = optimizers[name]
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_loss_dict[name].append(train_loss)

        # Test the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracy_dict[name].append(accuracy)

        schedulers[name].step()

# Plotting Training Loss and Test Accuracy
plt.figure(figsize=(12, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
for name, losses in train_loss_dict.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.legend()

# Plot Test Accuracy
plt.subplot(1, 2, 2)
for name, accuracies in test_accuracy_dict.items():
    plt.plot(accuracies, label=name)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
