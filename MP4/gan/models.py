import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1),  # 1 x 3 x 3
        )

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########
       output = self.model(x)
       return output.view(-1, 1).squeeze(1)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1, padding=0),  # 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),  # 3 x 64 x 64
            nn.Tanh()
        )

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

        return self.model(x)
