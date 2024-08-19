import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings  # For suppressing warnings
import pandas as pd
warnings.filterwarnings("ignore")  # Not recommended for production 

# Define the dataset class (replace with your actual dataset loading logic)
class RestaurantDataset(torch.utils.data.Dataset):
  def __init__(self, data_path, image_loader_fn):
    self.data_path = data_path
    self.image_loader_fn =image_loader_fn
    # Load your restaurant data (cuisine and image paths) from data_path
    # ... (data loading logic) ...
    data = pd.read_csv('cleaned_file.csv')
    return data

  def __len__(self):
    return len(self.data)  # Replace with the actual data length

  def __getitem__(self, idx):
    # Load image and cuisine label based on index
    image = load_image(self.data[idx]["image_path"])  # Replace with your image loading function
    cuisine = self.data[idx]["cuisine"]
    if self.image_transform:
      image = self.image_transform(image)
    return image, cuisine

# Define the text preprocessing function (replace with your specific logic)
def preprocess_text(text):
  # Convert text to lowercase
  text = text.lower()
  # Remove punctuation and special characters (optional)
  # ... (punctuation removal logic) ...
  return text

# Text Embedding Model
class Embedding(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(Embedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)

  def forward(self, text):
    # Preprocess text if needed (call the preprocess_text function)
    text = preprocess_text(text)
    # Convert text to numerical representation (e.g., one-hot encoding or tokenization)
    text_ids = convert_text_to_ids(text)  # Replace with your text to id conversion logic
    return self.embedding(text_ids)

# Generator Model
class Generator(nn.Module):
  def __init__(self, channels, noise_dim, embed_dim, embed_out_dim):
    super(Generator, self).__init__()
    self.channels = channels
    self.noise_dim = noise_dim
    self.embed_dim = embed_dim
    self.embed_out_dim = embed_out_dim

    self.text_embedding = Embedding(vocab_size, embed_dim)  # Replace vocab_size with actual value

    self.model = nn.Sequential(
        nn.ConvTranspose2d(noise_dim + embed_out_dim, 512, kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, output_padding=1),
        nn.Tanh()
    )

  def forward(self, noise, cuisine):
    # Embed the cuisine text
    text_embedding = self.text_embedding(cuisine)
    # Concatenate noise and text embedding
    z = torch.cat([noise, text_embedding], dim=1)
    return self.model(z)


