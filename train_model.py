# This code demonstrates how to train a Texture Refinement model using a custom dataset. 
#The training process utilizes a ResNet18 as a base for feature extraction, combined with a custom Attention Module and Texture Refiner to enhance the details and textures of images. 
#The model aims to refine images by emphasizing texture details, making it especially useful for improving image quality in tasks that require high levels of visual detail. 
#The script includes the setup for a VGG-based Perceptual Loss to complement traditional MSE loss, aiming to capture and enhance texture in a more perceptually relevant manner. 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from typing import Tuple

# Dataset path - Update this path to your dataset location
dataset_path = "/path/to/your/images"

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the Attention Module
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attention_map = self.conv(x)
        attention_map = torch.sigmoid(attention_map)
        x = x * attention_map * self.gamma + x
        return x

# Define the Texture Refiner Model
class TextureRefinerCNN(nn.Module):
    def __init__(self):
        super(TextureRefinerCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-2])
        self.attention = AttentionModule(512)
        self.refiner = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.refiner(x)
        return x

# Define the VGG-based Perceptual Loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model.features[:31]
        self.layer_weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]

    def forward(self, input, target):
        input_features = self.get_features(input)
        target_features = self.get_features(target)
        loss = 0
        for i, (input_feat, target_feat) in enumerate(zip(input_features, target_features)):
            loss += self.layer_weights[i] * nn.functional.l1_loss(input_feat, target_feat)
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in [3, 8, 15, 22, 29]:
                features.append(x)
        return features

# Instantiate the model, loss functions, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextureRefinerCNN().to(device)
vgg_model = models.vgg19(pretrained=True).features.to(device)
vgg_model.eval()
perceptual_loss = VGGPerceptualLoss(vgg_model).to(device)
mse_loss = nn.MSELoss()
loss_weights = [1.0, 0.1]  # Adjust these weights as needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust the number of epochs based on your need
for epoch in range(num_epochs):
    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        mse_loss_value = mse_loss(outputs, inputs)
        perceptual_loss_value = perceptual_loss(outputs, inputs)
        total_loss = loss_weights[0] * mse_loss_value + loss_weights[1] * perceptual_loss_value

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {mse_loss_value.item():.4f}, Perceptual Loss: {perceptual_loss_value.item():.4f}')

torch.save(model.state_dict(), 'texture_refiner_model.pth')
print('Model trained and saved successfully.')
