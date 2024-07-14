import torchvision, torch
from torchvision import transforms, datasets
import os
import shutil
from torch import nn, optim

import numpy as np
import cv2
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from artist import Artist
from critic import Critic


class ImageGenerator:
    def __init__(self, device, render_path):
        self.artist = Artist(device).to(device)
        self.critic = Critic().to(device)
        self.device = device


        #save frequency
        self.save_frequency = 4

        #training
        self.lr = 0.00015
        self.criterion = nn.BCELoss()
        self.optimizer_artist = optim.Adam(self.artist.parameters(), lr = self.lr, betas=(0.5, 0.999))
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = self.lr/2, betas=(0.5, 0.999))
        # self.scheduler_artist = StepLR(self.optimizer_artist, step_size=4, gamma=0.1, verbose = True)
        # self.scheduler_critic = StepLR(self.optimizer_critic, step_size=4, gamma=0.1, verbose = True)
        self.labels = {"real": 1, "fake": 0}
        self.render_path = render_path

    def get_dataloader(self, root_path):
        self.dataset = datasets.ImageFolder(root=root_path, transform = self.critic.transform)
        self.dataloader = DataLoader(self.dataset, batch_size = 8, shuffle = True)
        
    def train(self, epochs = 50):
        image_saves = 4
        t = 0
        image_number = 0
        fixed_noises = []
        for i in range(image_saves):
            fixed_noises.append(torch.randn(1, self.artist.input_size, 1,1,device = self.device))
        for i in range(image_saves):
            try:
                os.mkdir(os.path.join(self.render_path, f'line{i}'))
            except:
                pass


        try:
            os.mkdir(os.path.join(self.render_path, 'random_line'))
        except:
            pass
    

        for epoch in range(epochs):

            size = len(self.dataloader)
            for batch_num, data in enumerate(self.dataloader, 0):
                print(f"epoch: {epoch+1}/{epochs}   batch: {batch_num+1}/{size}", end = "\r")
                self.critic.zero_grad()
                device_data = data[0].to(self.device)
                batch_size = device_data.size(0)
                label = torch.full((batch_size,), self.labels["real"], dtype=torch.float, device = self.device)
                output = self.critic(device_data).view(-1)
                critic_err_real = self.criterion(output, label)
                critic_err_real.backward()


                noise = torch.randn(batch_size, self.artist.input_size, 1,1, device = self.device)
                fake = self.artist(noise)
                label.fill_(self.labels["fake"])
                output = self.critic(fake.detach()).view(-1)
                critic_err_fake = self.criterion(output, label)
                critic_err_fake.backward()
                critic_loss = output.mean().item()
                critic_err = critic_err_fake + critic_err_real
                self.optimizer_critic.step()


                self.artist.zero_grad()
                label.fill_(self.labels["real"])
                output = self.critic(fake).view(-1)
                artist_err = self.criterion(output, label)
                artist_err.backward()
                artist_loss = output.mean().item()
                self.optimizer_artist.step()

                if batch_num % 10 == 0:
                    a_loss = round(artist_err.item(), 5)
                    c_loss = round(critic_err.item(), 5)
                    print(f"a_loss: {a_loss}    c_loss: {c_loss}                  ")
                    image_number += 1

                if t % self.save_frequency == 0:
                    for j in range(image_saves):
                        
                        self.artist.make_image(os.path.join(self.render_path, f"line{j}" ,f"image{image_number}.png"), curr_path = os.path.join(self.render_path, f'line{j}curr.png'), fixed = fixed_noises[j])
                    self.artist.make_image(os.path.join(self.render_path, "random_line", f"randimage{image_number}.png"), curr_path = os.path.join(self.render_path,'most_recent.png'))

                t+= 1
            # self.scheduler_artist.step()
            # self.scheduler_artist.step()
   
        torch.save(self.artist, "artist.pth")
        torch.save(self.critic, "critic.pth")



                

                

                

