# This is a texture refinement model leveraging a U-Net architecture and adversarial training to enhance image textures. 
#The model combines MSE loss for content fidelity and adversarial loss from a pre-trained VGG19 to produce images with realistic and detailed textures. 
#It's designed for applications where high-quality visual detail is crucial.

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

dataset_path = "/path/to/your/images"

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#Texture Refiner Model
class TextureRefinerUNet(nn.Module):
    def __init__(self):
        super(TextureRefinerUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 2, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the model, loss function, optimizer
model = TextureRefinerUNet().to('cuda')
criterion = nn.MSELoss()
adversarial_criterion = nn.BCEWithLogitsLoss()
adversary = models.vgg19(pretrained=True).features[:36].to('cuda')
adversary.eval()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, _ in dataloader:
        inputs = inputs.to('cuda')

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Adversarial loss
        real_features = adversary(inputs)
        fake_features = adversary(outputs)
        adversarial_loss = adversarial_criterion(fake_features, torch.zeros_like(fake_features))
        adversarial_loss += adversarial_criterion(real_features, torch.ones_like(real_features))

        total_loss = loss + 0.01 * adversarial_loss  # Adjust the weight for adversarial loss

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Adversarial Loss: {adversarial_loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'texture_refiner_model.pth')
