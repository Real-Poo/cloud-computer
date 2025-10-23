import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialEncoder(nn.Module):
    def __init__(self, in_channels):
        super(InitialEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),           
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        return e1, p1


def main():
    x = torch.randn(1, 1, 572, 572)
    encoder = InitialEncoder(in_channels=1)
    e1, p1 = encoder(x)

    print("Encoder1 Output Shape:", e1.shape)
    print("Pooled Output Shape:", p1.shape)

if __name__ == "__main__":
    main()