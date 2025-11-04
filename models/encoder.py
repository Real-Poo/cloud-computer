import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class InitialEncoder(nn.Module):
    def __init__(self, in_channels):
        super(InitialEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),           
            nn.ReLU(inplace=True)
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),           
            # nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        return e1, p1


def main():
    x = torch.randn(1, 1, 572, 572)
    encoder1 = InitialEncoder(in_channels=1)
    e1, p1 = encoder1(x)

    print("Encoder1 Output Shape:", e1.shape)
    print("Pooled Output Shape:", p1.shape)


    encoder2 = InitialEncoder(in_channels=64)
    e2, p2 = encoder2(p1)

    print("Encoder2 Output Shape:", e2.shape)
    print("Pooled Output Shape:", p2.shape)

    encoder3 = InitialEncoder(in_channels=64)
    e3, p3 = encoder3(p2)

    print("Encoder2 Output Shape:", e3.shape)
    print("Pooled Output Shape:", p3.shape)

if __name__ == "__main__":
    main()