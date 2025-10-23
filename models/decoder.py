import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckDecoder(nn.Module):
    def __init__(self):
        super(BottleneckDecoder, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        b = self.bottleneck(x)

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, skip), dim=1)
        d4 = self.decoder4(d4)

        return d4
    

def main():
    
    x = torch.randn(1, 512, 28, 28)
    skip = torch.randn(1, 512, 56, 56)

    bottleneck_decoder = BottleneckDecoder()
    output = bottleneck_decoder(x, skip)

    print("Bottleneck and Decoder Output Shape:", output.shape) 


if __name__ == "__main__":
    main()