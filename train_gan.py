import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: latent_dim x 1 x 1
            self._block(latent_dim, feature_maps * 8, 4, 1, 0),  # Output: feature_maps*8 x 4 x 4
            self._block(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # Output: feature_maps*4 x 8 x 8
            self._block(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # Output: feature_maps*2 x 16 x 16
            self._block(feature_maps * 2, feature_maps, 4, 2, 1),  # Output: feature_maps x 32 x 32
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1),  # Output: img_channels x 64 x 64
            nn.ConvTranspose2d(img_channels, img_channels, kernel_size=4, stride=2, padding=1),  # Final layer to scale to 128x128
            nn.Tanh()  # Output: img_channels x 128 x 128
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_maps):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: img_channels x 128 x 128
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),  # Output: feature_maps x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            self._block(feature_maps, feature_maps * 2, 4, 2, 1),  # Output: feature_maps*2 x 32 x 32
            self._block(feature_maps * 2, feature_maps * 4, 4, 2, 1),  # Output: feature_maps*4 x 16 x 16
            self._block(feature_maps * 4, feature_maps * 8, 4, 2, 1),  # Output: feature_maps*8 x 8 x 8
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),  # Output: 1 x 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.disc(x)


# Dataset and DataLoader
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as a dummy label

# Now use this custom dataset class
data_dir = './archive/img_align_celeba/img_align_celeba/'

# Define a transformation to preprocess the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the dataset using the custom dataset class
dataset = CelebADataset(root_dir=data_dir, transform=transform)

# Define a DataLoader to read images in batches
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Check the number of images
print(f'Total number of images: {len(dataset)}')

#####################################################
# Gradient penalty
#####################################################
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size, c, h, w = real_images.shape
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_images)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated.requires_grad_(True)
    interpolated_logits = discriminator(interpolated)
    grad_outputs = torch.ones_like(interpolated_logits)
    gradients = torch.autograd.grad(
        outputs=interpolated_logits,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp

#####################################################
# model training
#####################################################

# Training with WGAN-GP
lambda_gp = 10  # Gradient penalty coefficient
latent_dim = 100
img_channels = 3
feature_maps_disc = 64
feature_maps_gen = 128
lr_gen = 0.0002
lr_disc = 0.0002
batch_size = 128
epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(latent_dim, img_channels, feature_maps_gen).to(device)
disc = Discriminator(img_channels, feature_maps_disc).to(device)

opt_gen = optim.RMSprop(gen.parameters(), lr=lr_gen)
opt_disc = optim.RMSprop(disc.parameters(), lr=lr_disc)

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.size(0)

        # Train Discriminator
        for _ in range(5):
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            disc_fake = disc(fake.detach()).view(-1)
            gp = gradient_penalty(disc, real, fake)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

        # Train Generator
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake = gen(noise)
        output = disc(fake).view(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    if epoch % 1 == 0:
        save_image(fake[:25], f"output_epoch_{epoch}.png", nrow=5, normalize=True)

    print(f"Epoch [{epoch}/{epochs}]  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # Save models periodically
    if epoch % 10 == 0:
        torch.save(gen.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(disc.state_dict(), f"discriminator_epoch_{epoch}.pth")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.utils import save_image
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# from PIL import Image
# import os

# # Generator Network
# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_channels, feature_maps):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             # Input: latent_dim x 1 x 1
#             self._block(latent_dim, feature_maps * 8, 4, 1, 0),  # Output: feature_maps*8 x 4 x 4
#             self._block(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # Output: feature_maps*4 x 8 x 8
#             self._block(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # Output: feature_maps*2 x 16 x 16
#             self._block(feature_maps * 2, feature_maps, 4, 2, 1),  # Output: feature_maps x 32 x 32
#             nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1),  # Output: img_channels x 64 x 64
#             nn.ConvTranspose2d(img_channels, img_channels, kernel_size=4, stride=2, padding=1),  # Final layer to scale to 128x128
#             nn.Tanh()  # Output: img_channels x 128 x 128
#         )

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         return self.gen(x)

# # Discriminator Network
# class Discriminator(nn.Module):
#     def __init__(self, img_channels, feature_maps):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             # Input: img_channels x 128 x 128
#             nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),  # Output: feature_maps x 64 x 64
#             nn.LeakyReLU(0.2, inplace=True),
#             self._block(feature_maps, feature_maps * 2, 4, 2, 1),  # Output: feature_maps*2 x 32 x 32
#             self._block(feature_maps * 2, feature_maps * 4, 4, 2, 1),  # Output: feature_maps*4 x 16 x 16
#             self._block(feature_maps * 4, feature_maps * 8, 4, 2, 1),  # Output: feature_maps*8 x 8 x 8
#             nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),  # Output: 1 x 1 x 1
#             nn.Sigmoid()  # Output: scalar
#         )

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def forward(self, x):
#         return self.disc(x)

# # Dataset and DataLoader
# class CelebADataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_name)
#         if self.transform:
#             image = self.transform(image)
#         return image, 0  # Returning 0 as a dummy label

# # Now use this custom dataset class
# data_dir = './archive/img_align_celeba/img_align_celeba/'

# # Define a transformation to preprocess the images
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images to 128x128
#     transforms.ToTensor(),          # Convert images to PyTorch tensors
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
# ])

# # Load the dataset using the custom dataset class
# dataset = CelebADataset(root_dir=data_dir, transform=transform)

# # Define a DataLoader to read images in batches
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# # Check the number of images
# print(f'Total number of images: {len(dataset)}')

# #####################################################
# # Gradient penalty
# #####################################################
# def gradient_penalty(discriminator, real_images, fake_images):
#     batch_size, c, h, w = real_images.shape
#     epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
#     epsilon = epsilon.expand_as(real_images)
#     interpolated = epsilon * real_images + (1 - epsilon) * fake_images
#     interpolated.requires_grad_(True)
#     interpolated_logits = discriminator(interpolated)
#     grad_outputs = torch.ones_like(interpolated_logits)
#     gradients = torch.autograd.grad(
#         outputs=interpolated_logits,
#         inputs=interpolated,
#         grad_outputs=grad_outputs,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_norm = gradients.norm(2, dim=1)
#     gp = ((gradient_norm - 1) ** 2).mean()
#     return gp

# #####################################################
# # model training
# #####################################################

# # Training with WGAN-GP
# lambda_gp = 10  # Gradient penalty coefficient
# latent_dim = 100
# img_channels = 3
# feature_maps_disc = 16
# feature_maps_gen = 128
# lr_gen = 0.0001
# lr_disc = 0.0004
# batch_size = 64
# epochs = 50

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gen = Generator(latent_dim, img_channels, feature_maps_gen).to(device)
# disc = Discriminator(img_channels, feature_maps_disc).to(device)

# opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
# opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# for epoch in range(epochs):
#     for batch_idx, (real, _) in enumerate(dataloader):
#         real = real.to(device)
#         batch_size = real.size(0)
#         noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
#         fake = gen(noise)

#         # # Debugging: Print shapes to understand mismatch
#         # print(f"Real images shape: {real.shape}")
#         # print(f"Fake images shape: {fake.shape}")

#         # Train Discriminator with WGAN-GP
#         disc_real = disc(real).view(-1)
#         disc_fake = disc(fake.detach()).view(-1)
#         gp = gradient_penalty(disc, real, fake)

#         loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp
#         disc.zero_grad()
#         loss_disc.backward()
#         opt_disc.step()

#         # Train Generator
#         for _ in range(4):  # Experiment with training generator more times
#             noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
#             fake = gen(noise)
#             output = disc(fake).view(-1)
#             loss_gen = -torch.mean(output)
#             gen.zero_grad()
#             loss_gen.backward()
#             opt_gen.step()

#     if epoch % 1 == 0:
#         save_image(fake[:25], f"output_epoch_{epoch}.png", nrow=5, normalize=True)

#     print(f"Epoch [{epoch}/{epochs}]  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.utils import save_image
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# from PIL import Image
# import os


# # Generator Network
# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_channels, feature_maps):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             # Input: latent_dim x 1 x 1
#             self._block(latent_dim, feature_maps * 8, 4, 1, 0),  # Output: feature_maps*8 x 4 x 4
#             self._block(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # Output: feature_maps*4 x 8 x 8
#             self._block(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # Output: feature_maps*2 x 16 x 16
#             self._block(feature_maps * 2, feature_maps, 4, 2, 1),  # Output: feature_maps x 32 x 32
#             nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()  # Output: img_channels x 128 x 128
#         )

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         return self.gen(x)

# # Discriminator Network
# class Discriminator(nn.Module):
#     def __init__(self, img_channels, feature_maps):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             # Input: img_channels x 128 x 128
#             nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),  # Output: feature_maps x 64 x 64
#             nn.LeakyReLU(0.2, inplace=True),
#             self._block(feature_maps, feature_maps * 2, 4, 2, 1),  # Output: feature_maps*2 x 32 x 32
#             self._block(feature_maps * 2, feature_maps * 4, 4, 2, 1),  # Output: feature_maps*4 x 16 x 16
#             self._block(feature_maps * 4, feature_maps * 8, 4, 2, 1),  # Output: feature_maps*8 x 8 x 8
#             nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),  # Output: 1 x 1 x 1
#             nn.Sigmoid()  # Output: scalar
#         )

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def forward(self, x):
#         return self.disc(x)

# #####################################################
# # load in images
# #####################################################

# class CelebADataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_name)
#         if self.transform:
#             image = self.transform(image)
#         return image, 0  # Returning 0 as a dummy label

# # Now use this custom dataset class
# data_dir = './archive/img_align_celeba/img_align_celeba/'

# # Define a transformation to preprocess the images
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images to 128x128
#     transforms.ToTensor(),          # Convert images to PyTorch tensors
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
# ])

# # Load the dataset using the custom dataset class
# dataset = CelebADataset(root_dir=data_dir, transform=transform)

# # Define a DataLoader to read images in batches
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# # Check the number of images
# print(f'Total number of images: {len(dataset)}')


# #####################################################
# # Gradient penalty
# #####################################################
# def gradient_penalty(discriminator, real_images, fake_images):
#     batch_size, c, h, w = real_images.shape
#     # Ensure epsilon is broadcast correctly across all dimensions
#     epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
#     epsilon = epsilon.expand_as(real_images)  # Match the size of real_images and fake_images
#     interpolated = epsilon * real_images + (1 - epsilon) * fake_images
#     interpolated.requires_grad_(True)
#     interpolated_logits = discriminator(interpolated)
#     grad_outputs = torch.ones_like(interpolated_logits)
#     gradients = torch.autograd.grad(
#         outputs=interpolated_logits,
#         inputs=interpolated,
#         grad_outputs=grad_outputs,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_norm = gradients.norm(2, dim=1)
#     gp = ((gradient_norm - 1) ** 2).mean()
#     return gp



# #####################################################
# # model training
# #####################################################

# # Training with WGAN-GP
# lambda_gp = 10  # Gradient penalty coefficient
# latent_dim = 100
# img_channels = 3
# feature_maps_disc = 16
# feature_maps_gen = 128
# lr_gen = 0.0001
# lr_disc = 0.0004
# batch_size = 64
# epochs = 50

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gen = Generator(latent_dim, img_channels, feature_maps_gen).to(device)
# disc = Discriminator(img_channels, feature_maps_disc).to(device)

# opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
# opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# for epoch in range(epochs):
#     for batch_idx, (real, _) in enumerate(dataloader):
#         real = real.to(device)
#         batch_size = real.size(0)
#         noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
#         fake = gen(noise)

#         # Train Discriminator with WGAN-GP
#         disc_real = disc(real).view(-1)
#         disc_fake = disc(fake.detach()).view(-1)
#         gp = gradient_penalty(disc, real, fake)

#         loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp
#         disc.zero_grad()
#         loss_disc.backward()
#         opt_disc.step()

#         # Train Generator
#         for _ in range(4):  # Experiment with training generator more times
#             noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
#             fake = gen(noise)
#             output = disc(fake).view(-1)
#             loss_gen = -torch.mean(output)
#             gen.zero_grad()
#             loss_gen.backward()
#             opt_gen.step()

#     if epoch % 1 == 0:
#         save_image(fake[:25], f"output_epoch_{epoch}.png", nrow=5, normalize=True)

#     print(f"Epoch [{epoch}/{epochs}]  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")


# # # Hyperparameters
# # latent_dim = 100
# # img_channels = 3
# # feature_maps_disc = 16
# # feature_maps_gen = 128
# # lr_gen = 0.0007
# # lr_disc = 0.0001
# # batch_size = 64
# # epochs = 50

# # # Initialize generator and discriminator
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # gen = Generator(latent_dim, img_channels, feature_maps_gen).to(device)
# # disc = Discriminator(img_channels, feature_maps_disc).to(device)

# # # Optimizers
# # opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
# # opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# # # Loss function
# # criterion = nn.BCELoss()

# # # Training loop
# # for epoch in range(epochs):
# #     for batch_idx, (real, _) in enumerate(dataloader):
# #         real = real.to(device)
# #         batch_size = real.size(0)
# #         noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        
# #         # Generate real and fake labels with smoothing
# #         real_labels = torch.full((batch_size,), 0.95, dtype=torch.float, device=device)  # Smoothed real labels
# #         fake_labels = torch.full((batch_size,), 0.05, dtype=torch.float, device=device)  # Smoothed fake labels

# #         # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
# #         fake = gen(noise)
# #         disc_real = disc(real).view(-1)
        
# #         # Correct label size to match discriminator output
# #         real_labels = torch.full_like(disc_real, 0.95, dtype=torch.float, device=device)  # Matching size with disc_real
# #         loss_disc_real = criterion(disc_real, real_labels)
        
# #         disc_fake = disc(fake.detach()).view(-1)
# #         fake_labels = torch.full_like(disc_fake, 0.05, dtype=torch.float, device=device)  # Matching size with disc_fake
# #         loss_disc_fake = criterion(disc_fake, fake_labels)
        
# #         loss_disc = (loss_disc_real + loss_disc_fake) / 2
# #         disc.zero_grad()
# #         loss_disc.backward()
# #         opt_disc.step()

# #         # Train Generator multiple times for each discriminator update
# #         for _ in range(4):  # Experiment with different numbers, e.g., 1, 2, 3, etc.
# #             noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
# #             fake = gen(noise)
# #             output = disc(fake).view(-1)
            
# #             # Matching real_labels to generator output size
# #             real_labels = torch.full_like(output, 0.95, dtype=torch.float, device=device)  # Matching size with output
# #             loss_gen = criterion(output, real_labels)
            
# #             gen.zero_grad()
# #             loss_gen.backward()
# #             opt_gen.step()

# #     # Save some generated samples for inspection
# #     if epoch % 4 == 0:
# #         save_image(fake[:25], f"output_{epoch}.png", nrow=5, normalize=True)

# #     print(f"Epoch [{epoch}/{epochs}]  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")


