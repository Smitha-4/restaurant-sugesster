
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

class Generator(nn.Module):
    def __init__(self, channels, noise_dim=100, embed_dim=1024, embed_out_dim=128):
        super(Generator, self).__init__()
        self.channels=channels
        self.noise_dim=noise_dim
        self.embed_dim=embed_dim
        self.embed_out_dim= embed_out_dim

        self.text_embedding=nn.Sequential(nn.Linear(self.embed_dim, self.embed_out_dim),
                                          nn.BatchNorm1d(self.embed_out_dim),
                                          nn.LeakyReLU(0.2, inplace=True))
        model=[]
        model += self.create_layer(self.noise + self.embed_out_dim, 512, 4, stride=1, padding=0)
        model += self.create_layer(512,256,4, stride=2, padding=1)
        model += self.create_layer(256, 128, 4, stride=2, padding=1)
        model += self.create_layer(128, 64, 4, stride=2, padding=1)
        model += self.create_layer(64, self.channels, 4, stride=2, padding=1, output=True)
        self.model=nn.Sequential(*model)
    def create_layer(self, size_in, size_out, kernel_size = 4, stride=2, padding=1, output = False):
        layers=[nn.ConvTranspose2d(size_in, size_out, kernel_size, stride=stride, padding=padding, bias=False)]
        if output:
            layers.append(nn.Tanh())
        else:
            layers += [nn.BatchNorm2d(size_out), nn.ReLU(True)]
        return layers
    
    def forward(self, noise, text):
        #Apply the text embedding to the input text
        text = self.text_embedding(text)
        text = text.view(text.shape)
        z= torch.cat([text, noise], 1)# Concatenate the text with the noise

        return self.model(z)
    

#Embedding the model
class Embedding(nn.Module):
    def __init__(self, size_in, size_out) :
        super(Embedding, self).__init__()
        self.text_embedding =nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, text):
        embed_out=self.text_embedding(text)
        embed_out_resize =embed_out.repeat(4,4,1,1).permute(2,3,0,1)
        out =torch.cat([x, embed_out_resize],1)# concatenate the text embedding
        return out
    
# Discriminator model
    
class Discriminator(nn.Module):
    def __init__(self, channels, embed_dim =1024, embed_out_dim =128):
        super(Discriminator, self).__init__()
        self.channels =channels
        self.embed_dim =embed_dim
        self.embed_out_dim =embed_out_dim

        # Discriminator architecture

        self.model =nn.Sequential(
            *self.create_layer(self.channels, 64,4,2,1, normalize=False),
            *self.create_layer(64,128,4,2,1),
            *self.create_layer(128,256,4,2,1),
            *self.create_layer(256,512,4,2,1),
        )
        self.text_embedding =Embedding(self.embed_dim, self.embed_out_dim)
        self.output =nn.Sequential(
            nn.Conv2d(512 + self.embed_out_dim, 1,4,1,0, bias =False), nn.Sigmoid()
        )
    def create_layer(self, size_in, size_out, kernel_size=4,stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(size_in, size_out, kernel_size=kernel_size, stride=stride, padding=padding)]
        if normalize:
            layers.append(nn.BatchNorm2d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    def forward(self, x, text):
        x_out =self.model(x)
        out =self.text_embedding(x_out, text)
        out =self.output(out)
        return out.squeeze(), x_out

class RestaurantDataset(torch._utils.data.Dataset):
    def __init__(self, data_path, image_transform=None):
        self.data_path =data_path
        self.image_transform=image_transform
        # Loading the dataset
        data=pd.read_csv('cleaned_file.csv', )

