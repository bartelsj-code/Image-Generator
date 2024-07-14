import torchvision, torch
from torchvision import transforms, datasets
import os
import shutil
from torch import nn, optim
import numpy as np
import cv2
from PIL import Image



class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.output_size = 128

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.output_size, 4, 2, 1, bias = False),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.output_size, self.output_size * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.output_size * 2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.output_size*2, self.output_size * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.output_size * 4),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.output_size*4, self.output_size * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.output_size * 8),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(self.output_size*8, self.output_size * 16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.output_size * 16),
            nn.ReLU(True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(self.output_size*16, self.output_size * 16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.output_size * 16),
            nn.ReLU(True)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(self.output_size*16, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )


    def forward(self, x):
        
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        # print(out.size())
        out = self.layer7(out)
        # print(out.size())

        return out

    def prep_image(self, img):
        tensor = self.transform(img).unsqueeze(0)
        return tensor
        
# if __name__ == "__main__":
#     critic = Critic()

    
#     image_file = "renderings/image1.png"
#     img = Image.open(image_file)
    
#     prepped = critic.prep_image(img)
#     critic.eval()
#     critic(prepped)

#     pass
    