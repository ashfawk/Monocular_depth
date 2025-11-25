import torch
import torch.nn as nn
import torchvision.models as models

class FastDepth(nn.Module):
    def __init__(self, pretrained=True):
        super(FastDepth, self).__init__()

        # Load MobileNetV2 as Encoder
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.encoder = mobilenet.features

        # Decoder (Upsampling Layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(160, 80, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(80, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)  # Encode features
        x = self.decoder(x)  # Decode to depth map
        return x

# Test model initialization
if __name__ == "__main__":
    model = FastDepth()
    x = torch.randn(1, 3, 224, 224)  # Sample input
    output = model(x)
    print("Output Shape:", output.shape)  # Expected: (1, 1, 224, 224)
