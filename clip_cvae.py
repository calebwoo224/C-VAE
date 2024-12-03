import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Class names for FashionMNIST labels
fashion_mnist_classes = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot"    # 9
]

class CustomFashionMNIST(Dataset):
    def __init__(self, root, train=True, transform1=None, transform2=None, include_text=False):
        """
        Args:
            root (str): Path to download and store the FashionMNIST dataset.
            train (bool): If True, load the training dataset; otherwise, load the test dataset.
            transform1 (callable, optional): The first set of transforms to apply to the images.
            transform2 (callable, optional): The second set of transforms to apply to the images.
            include_text (bool): If True, include text descriptions based on the class labels.
        """
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True
        )
        self.transform1 = transform1
        self.transform2 = transform2
        self.include_text = include_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            image1 (Tensor): Image after the first transformation.
            image2 (Tensor): Image after the second transformation.
            label (int): The class label.
            text (str, optional): Text description of the class (if include_text=True).
        """
        image, label = self.dataset[idx]

        # Apply the first transformation
        if self.transform1:
            image1 = self.transform1(image)
        else:
            image1 = image

        # Apply the second transformation
        if self.transform2:
            image2 = self.transform2(image)
        else:
            image2 = image

        # Include text description if required
        if self.include_text:
            text = fashion_mnist_classes[label]
            return image1, image2, label, text

        return image1, image2, label

# Define transformations
transform_preprocess = preprocess  # For CLIP image encoding
transform_flatten = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])

# Initialize custom dataset
train_dataset = CustomFashionMNIST(
    root="./data",
    train=True,
    transform1=transform_preprocess,
    transform2=transform_flatten,
    include_text=True  # Include text descriptions
)

test_dataset = CustomFashionMNIST(
    root="./data",
    train=False,  # Use test split for testing
    transform1=transform_preprocess,
    transform2=transform_flatten,
    include_text=True
)

# Determine optimal batch size based on number of GPUs
base_batch_size = 128  # Original batch size
adjusted_batch_size = base_batch_size * max(1, num_gpus)

train_loader = DataLoader(
    train_dataset,
    batch_size=adjusted_batch_size,
    shuffle=True,
    num_workers=4,  # Adjust based on your system
    pin_memory=True if device == "cuda" else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=adjusted_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True if device == "cuda" else False
)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # Reconstruction loss
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Define the Conditional VAE
class CVAE(nn.Module):
    def __init__(self, n_in, n_hid, z_dim, clip_embed_dim=512):
        super(CVAE, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(n_in + clip_embed_dim, n_hid)
        self.fc21 = nn.Linear(n_hid, z_dim)  # For mean
        self.fc22 = nn.Linear(n_hid, z_dim)  # For log variance

        # Decoder layers
        self.fc3 = nn.Linear(z_dim + clip_embed_dim, n_hid)
        self.fc4 = nn.Linear(n_hid, n_in)

    def encode(self, x, c):
        """Encoder forward pass with conditioning."""
        # Concatenate input with conditioning vector
        x = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample epsilon from standard normal
        return mu + eps * std

    def decode(self, z, c):
        """Decoder forward pass with conditioning."""
        # Concatenate latent variable with conditioning vector
        z = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Assuming input images are normalized between 0 and 1

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

# Hyperparameters
learning_rate = 1e-3
n_epochs = 30
n_in = 28 * 28
z_dim = 40
n_hid = 800
clip_embed_dim = 512

model = CVAE(n_in=n_in, n_hid=n_hid, z_dim=z_dim, clip_embed_dim=clip_embed_dim).to(device)

# Wrap the model with DataParallel if multiple GPUs are available
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for training.")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create directory for saving models
os.makedirs('models', exist_ok=True)

train_losses = []
val_losses = []

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, (clip_img, og_img, label, text) in enumerate(train_loader):
        clip_img = clip_img.to(device)  # Images for CLIP encoding
        og_img = og_img.to(device)      # Flattened original images
        label = label.to(device)
        text_tokens = clip.tokenize(text).to(device)

        # Encode text descriptions using CLIP's text encoder
        
        with torch.no_grad():
            text_embed = clip_model.encode_text(text_tokens).float()

        # Normalize the text embeddings
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(og_img, text_embed)
        loss = loss_function(recon_batch, og_img, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    average_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(average_train_loss)
    print(f"Epoch: {epoch}, Train Loss: {average_train_loss:.4f}")

    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (clip_img, og_img, label, text) in enumerate(test_loader):
            clip_img = clip_img.to(device)
            og_img = og_img.to(device)
            label = label.to(device)

            # Encode text descriptions using CLIP's text encoder
            text_tokens = clip.tokenize(text).to(device)
            text_embed = clip_model.encode_text(text_tokens).float()
            text_embed /= text_embed.norm(dim=-1, keepdim=True)

            recon_batch, mu, logvar = model(og_img, text_embed)
            val_loss += loss_function(recon_batch, og_img, mu, logvar).item()

    average_val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(average_val_loss)
    print(f"Epoch: {epoch}, Validation Loss: {average_val_loss:.4f}")

    # # Save checkpoint after each epoch
    # checkpoint_path = f'models/cvae_fashion_mnist_epoch_{epoch}.pth'
    # if num_gpus > 1:
    #     torch.save(model.module.state_dict(), checkpoint_path)
    # else:
    #     torch.save(model.state_dict(), checkpoint_path)
    # print(f"Checkpoint saved to {checkpoint_path}")

# Save the final model
final_model_path = 'models/cvae_fashion_mnist_final.pth'
if num_gpus > 1:
    torch.save(model.module.state_dict(), final_model_path)
else:
    torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

