# This script generates images based on textual prompts using the SDXL and ByteDance Lightning model, then refines these images with a pre-trained Texture Refiner CNN to enhance detail and texture quality. 
# Please ensure the Texture Refiner CNN is trained before use.
 

import torch
from torch import nn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet18
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from IPython.display import display

#Attention Module
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

#Texture Refiner Model
class TextureRefinerCNN(nn.Module):
    def __init__(self):
        super(TextureRefinerCNN, self).__init__()

        base_model = resnet18(pretrained=True)
        self.features = create_feature_extractor(base_model, return_nodes={'layer4': 'out'})

        self.attention = AttentionModule(512)

        self.refiner = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.features(x)['out']
        x = self.attention(x)
        x = self.refiner(x)
        return x
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"

unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
term = input("Enter your prompt: ")
prompt = f"Generate a 4k, ((full-body:2)) image of a ((single:2)) {term} suitable for 3D modeling: with true colors of {term}, under uniform studio lighting, against a solid color background. The image should be ((single:2)) and centered, with its entire structure visible with smooth and accurately colored textures."
image = pipe(prompt, num_inference_steps=5, guidance_scale=2.5).images[0]
display(image)

# Transform the generated image for texture refinement
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).to('cuda')  # Add batch dimension and send to CUDA
model = TextureRefinerCNN().to('cuda')
model.load_state_dict(torch.load('texture_refiner_model.pth'))
refined_image_tensor = model(image_tensor)
refined_image = transforms.ToPILImage()(refined_image_tensor.squeeze(0).cpu())
display(refined_image)
