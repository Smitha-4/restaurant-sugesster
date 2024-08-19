import torch
import torch.nn as nn

#Generator Model
class Generator(nn.Module):
    def __init__(self, channels, noise_dimension = 100, embed_dimension = 1024, embed_output_dimension = 128):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dimension = noise_dimension
        self.embed_dimension = embed_dimension
        self.embed_output_dim = embed_output_dimension
        self.text_embedding = nn.Sequential(nn.Linear(self.embed_dimension, self.embed_output_dim),
                                            nn.BatchNorm1d(self.embed_output_dim),
                                            nn.LeakyReLU(0.2, inplace = True))
        #Generator Architecture
        model = []
        model += self.create_layer(self.noise_dim + self.embed_output_dim, 512,4, stripe=1,padding = 0)
        model += self.create_layer(512,256,4,stride=2, padding = 1)
        model += self.create_layer(256,128,4,stride = 2, padding=1)
        model += self.create_layer(128,64,4,stride=2, padding = 1, output =True)
        self.model = nn.Sequential(*model)

    def create_layer(self, in_size, out_size, kernel_size = 4, stride =2, padding =1, output =False):
        layers = [nn.ConvTranspose2d(in_size,out_size, kernel_size, stride =stride, padding = padding, bias = False)]
        if output:
            layers.append(nn.Tanh())
        else:
            layers += [nn.BatchNorm2d(out_size), nn.ReLU(True)]
        return layers
    def forward(self, noise, text):
        text = self.text_embedding(text)
        text =text.view(text.shape[0], text.shape[1],1,1)
        z = torch.cat([text, noise],1)
        return self.model(z)
        
# The Embedding model
class Embedding(nn.Module):
    def __init__(self, in_size, out_size):
        super(Embedding, self).__init__()
        self.text_embedding = nn.Sequential( 
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.LeakyReLU( 0.2, inplace=True))
    def forward(self, x, text):
        embed_out = self.text_embedding(text)
        embed_out_resize = embed_out.repeat(4,4,1,1).permute(2,3,0,1)
        out = torch.cat([x, embed_out_resize], 1)
        return out
#The Discriminator model
class Discriminator(nn.Module):
    def __init__(self, channels, embed_dimension = 1024, embed_output_dimension = 128):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dimension = embed_dimension
        self.embed_output_dimension =self.embed_output_dimension

        #Dicriminator Architecture
        self.model = nn.Sequential(
            *self.create_layer(self, channels, 64,4,2,1, normalize = False),
            *self.create_layers(64,128, 4,2,1)
            *self.create_layers(128, 256,4,2,1)
            *self.create_layers(256, 512,4,2,1)
        )
        self.text_embediing = Embedding(self.embed_dimension, self.embed_output_dimension)
        self.output = nn.Sequential(nn.Conv2D(512, self.embed_output_dimension, 1,4,1,0, bias =False),
                                    nn.Sigmoid)
        
    def create_layer(self, in_size, out_size, kernel_size = 4, stride= 2, padding = 1, normalize =True):
        layers = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride =stride, padding=padding)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        return layers
    def forward(self, x, text):
        x_out = self.model(x)
        out = self.text_embediing(x_out, text)
        out = self.output(out)
        return out.squeeze(), x_out