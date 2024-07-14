import torchvision, torch
from torchvision import transforms, datasets
import os
import shutil
from torch import nn, optim
import numpy as np
import cv2
from PIL import Image


class Artist(nn.Module):
    def __init__(self, device):
        super(Artist, self).__init__()
        self.input_size = 100
        self.output_dim = 256
        self.device = device
        self.image_transform = transforms.ToPILImage()

        self.layer_scalars = [16,16,8,4,2,1]
        self.transpose_layer1 = nn.Sequential(
            nn.ConvTranspose2d(self.input_size, self.output_dim * self.layer_scalars[0], 4, 1, 0, bias = False),
            nn.BatchNorm2d(self.output_dim*self.layer_scalars[0]),
            nn.ReLU(True)
        )
        self.transpose_layer2 = nn.Sequential(
            nn.ConvTranspose2d(self.output_dim*self.layer_scalars[0], self.output_dim*self.layer_scalars[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.output_dim*self.layer_scalars[1]),
            nn.ReLU(True)
        )
        self.transpose_layer3 = nn.Sequential(
            nn.ConvTranspose2d(self.output_dim*self.layer_scalars[1], self.output_dim*self.layer_scalars[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.output_dim*self.layer_scalars[2]),
            nn.ReLU(True)
        )
        self.transpose_layer4 = nn.Sequential(
            nn.ConvTranspose2d(self.output_dim*self.layer_scalars[2], self.output_dim*self.layer_scalars[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.output_dim*self.layer_scalars[3]),
            nn.ReLU(True)
        )
        self.transpose_layer5 = nn.Sequential(
            nn.ConvTranspose2d(self.output_dim*self.layer_scalars[3], self.output_dim*self.layer_scalars[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.output_dim*self.layer_scalars[4]),
            nn.ReLU(True)
        )
        self.transpose_layer6 = nn.Sequential(
            nn.ConvTranspose2d(self.output_dim*self.layer_scalars[4], self.output_dim*self.layer_scalars[5], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.output_dim*self.layer_scalars[5]),
            nn.ReLU(True)
        )
        self.transpose_layer7 = nn.Sequential(
            nn.ConvTranspose2d(self.output_dim*self.layer_scalars[5], 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def forward(self, x):

        out = self.transpose_layer1(x)
        # print(out.size())
        out = self.transpose_layer2(out)
        # print(out.size())
        out = self.transpose_layer3(out)
        # print(out.size())
        out = self.transpose_layer4(out)
        # print(out.size())
        out = self.transpose_layer5(out)
        # print(out.size())
        out = self.transpose_layer6(out)
        # print(out.size())
        out = self.transpose_layer7(out)
        # print(out.size())
        out = self.normalizer(out)
        return out

    def make_image(self, image_path, curr_path = None, fixed = None):
        with torch.no_grad():
            if fixed != None:
                x = fixed
            else:
                x = torch.randn(1, self.input_size, 1,1, device = self.device)
            #print(x.size())
            # out = self.linear_layers(x)
 
            # out = out.unsqueeze(2)
            # out = out.unsqueeze(3)

            out = self.transpose_layer1(x)
            #print(out.size())
            out = self.transpose_layer2(out)
            #print(out.size())
            out = self.transpose_layer3(out)
            #print(out.size())
            out = self.transpose_layer4(out)
            #print(out.size())
            out = self.transpose_layer5(out)
            #print(out.size())
            out = self.transpose_layer6(out)
            #print(out.size()) 
            out = self.transpose_layer7(out)

        out = (1+out)/2
        out = out.detach().cpu()
        img = out.squeeze()
        img = self.image_transform(img)

        img.save(image_path)
        if curr_path != None:
            img.save(curr_path)
        


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    artist = Artist(device)
    artist.eval()
    data = torch.randn(1, artist.input_size, 1, 1)
    img = artist(data).squeeze()
    rgb_image = cv2.cvtColor(((img)*255).detach().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image.astype('uint8'))
    pil_image.save('renderings/image1.png')