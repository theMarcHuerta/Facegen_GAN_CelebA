import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

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
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1),
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
            nn.Sigmoid()  # Output: scalar
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.disc(x)

# Hyperparameters
latent_dim = 100
img_channels = 3
feature_maps = 64
lr = 0.0002
batch_size = 64
epochs = 50

# Initialize generator and discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(latent_dim, img_channels, feature_maps).to(device)
disc = Discriminator(img_channels, feature_maps).to(device)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Save some generated samples for inspection
    if epoch % 10 == 0:
        save_image(fake[:25], f"output_{epoch}.png", nrow=5, normalize=True)

    print(f"Epoch [{epoch}/{epochs}]  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
