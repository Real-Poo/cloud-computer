import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image

class Cloud(nn.Module):
    def __init__(self):
        super(Cloud, self).__init__()

        def C2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        # down        
        self.e1_1 = C2d(in_channels=1, out_channels=64)
        self.e1_2 = C2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.e2_1 = C2d(in_channels=64, out_channels=128)
        self.e2_2 = C2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.e3_1 = C2d(in_channels=128, out_channels=256)
        self.e3_2 = C2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.e4_1 = C2d(in_channels=256, out_channels=512)
        self.e4_2 = C2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.e5_1 = C2d(in_channels=512, out_channels=1024)
        
        # up
        self.d5_1 = C2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.d4_2 = C2d(in_channels=2*512, out_channels=512)
        self.d4_1 = C2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.d3_2 = C2d(in_channels=2*256, out_channels=256)
        self.d3_1 = C2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.d2_2 = C2d(in_channels=2*128, out_channels=128)
        self.d2_1 = C2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.d1_2 = C2d(in_channels=2*64, out_channels=64)
        self.d1_1 = C2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # 업샘플 결과 x를 ref의 공간 크기에 정확히 맞춤
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # encoder
        e1_1 = self.e1_1(x)
        e1_2 = self.e1_2(e1_1)
        p1   = self.pool1(e1_2)

        e2_1 = self.e2_1(p1)
        e2_2 = self.e2_2(e2_1)
        p2   = self.pool2(e2_2)

        e3_1 = self.e3_1(p2)
        e3_2 = self.e3_2(e3_1)
        p3   = self.pool3(e3_2)   # <-- fix

        e4_1 = self.e4_1(p3)
        e4_2 = self.e4_2(e4_1)
        p4   = self.pool4(e4_2)   # <-- fix

        e5_1 = self.e5_1(p4)

        # decoder
        d5_1 = self.d5_1(e5_1)

        u4   = self.unpool4(d5_1)
        u4   = self._match_size(u4, e4_2)
        d4_2 = self.d4_2(torch.cat((u4, e4_2), dim=1))
        d4_1 = self.d4_1(d4_2)

        u3   = self.unpool3(d4_1)
        u3   = self._match_size(u3, e3_2)
        d3_2 = self.d3_2(torch.cat((u3, e3_2), dim=1))  # <-- use d3_*
        d3_1 = self.d3_1(d3_2)

        u2   = self.unpool2(d3_1)
        u2   = self._match_size(u2, e2_2)
        d2_2 = self.d2_2(torch.cat((u2, e2_2), dim=1))  # <-- use d2_*
        d2_1 = self.d2_1(d2_2)

        u1   = self.unpool1(d2_1)
        u1   = self._match_size(u1, e1_2)
        d1_2 = self.d1_2(torch.cat((u1, e1_2), dim=1))  # <-- use d1_*
        d1_1 = self.d1_1(d1_2)

        out  = self.fc(d1_1)
        return out

    

def main():
    image = cv2.imread('test.png')

    if image is None:
        print("Error: Could not load image")
        return
    

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('origin image', gray_image)
    cv2.waitKey(3)
    cloud_model = Cloud()

    input_array = gray_image.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_array)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    cloud_model.to(device)
    cloud_model.eval()

    with torch.no_grad():
        logits = cloud_model(input_tensor)
        prob = torch.softmax(logits, dim=1)[:, 1]  # 전경(class=1) 확률
    prob_np = (prob.squeeze().cpu().numpy()*255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_np, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
    cv2.imshow('FG probability heatmap', overlay)
    cv2.waitKey(0)


    # with torch.no_grad():
    #     output_tensor = cloud_model(input_tensor)

    # probabilities = F.softmax(output_tensor, dim=1)
    
    # predicted_mask_tensor = torch.argmax(probabilities, dim=1)
    
    # predicted_mask_array = predicted_mask_tensor.squeeze().cpu().numpy()
    
    # result = (predicted_mask_array * 255).astype(np.uint8)

    # print('logits shape:', output_tensor.shape)
    # print('unique mask values:', np.unique(predicted_mask_array, return_counts=True))

    # cv2.imshow('Predicted Mask', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()