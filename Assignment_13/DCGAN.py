import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Configuration dictionary for easy parameter management
CONFIG = {
    'workers': 4,
    'data_dir': './datasets/CelebA',
    'image_size': 64,
    'channels': 3,
    'batch_size': 256,
    'latent_dim': 100,
    'epochs': 50,
    'learning_rate': 0.0002,
    'adam_beta_1': 0.5,
    'real_label': 1.0,
    'fake_label': 0.0,
    'generator_path': 'dcgan_generator.pth',
    'discriminator_path': 'dcgan_discriminator.pth',
    'output_dir': 'gan_images_pytorch'
}

def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def create_dataloader(config):
    """Create and return the dataloader for training."""
    transforms_list = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = dset.ImageFolder(root=config['data_dir'], transform=transforms_list)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['workers']
    )
    return dataloader

def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def create_generator(latent_dim, channels):
    """Create and return the Generator model."""
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)
    
    return Generator()

def create_discriminator(channels):
    """Create and return the Discriminator model."""
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)
    
    return Discriminator()

def initialize_models(config, device):
    """Initialize and return the Generator and Discriminator models."""
    netG = create_generator(config['latent_dim'], config['channels']).to(device)
    netD = create_discriminator(config['channels']).to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print("Generator Architecture:")
    print(netG)
    print("\nDiscriminator Architecture:")
    print(netD)
    
    return netG, netD

def create_optimizers(netG, netD, config):
    """Create and return optimizers for both networks."""
    optimizerD = optim.Adam(
        netD.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['adam_beta_1'], 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['adam_beta_1'], 0.999)
    )
    return optimizerG, optimizerD

def create_fixed_noise(config, device):
    """Create fixed noise for monitoring training progress."""
    return torch.randn(64, config['latent_dim'], 1, 1, device=device)

def setup_training_environment(config):
    """Setup directories and return training components."""
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    
    return nn.BCELoss(), [], [], []

def train_discriminator(netD, netG, data, criterion, optimizerD, config, device):
    """Train the discriminator for one batch."""
    netD.zero_grad()
    
    # Train on real data
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), config['real_label'], dtype=torch.float, device=device)
    
    output = netD(real_cpu).view(-1)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    # Train on fake data
    noise = torch.randn(b_size, config['latent_dim'], 1, 1, device=device)
    fake = netG(noise)
    label.fill_(config['fake_label'])
    
    output = netD(fake.detach()).view(-1)
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    
    errD = errD_real + errD_fake
    optimizerD.step()
    
    return errD, fake, D_x, D_G_z1

def train_generator(netD, netG, fake, criterion, optimizerG, config, device):
    """Train the generator for one batch."""
    netG.zero_grad()
    
    b_size = fake.size(0)
    label = torch.full((b_size,), config['real_label'], dtype=torch.float, device=device)
    
    output = netD(fake).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()
    
    return errG, D_G_z2

def save_epoch_images(netG, fixed_noise, epoch, config):
    """Generate and save images for the current epoch."""
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    
    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
    vutils.save_image(img_grid, f"{config['output_dir']}/epoch_{epoch+1}.png")
    return img_grid

def train_gan(netG, netD, dataloader, config, device):
    """Main training loop for the GAN."""
    criterion, img_list, G_losses, D_losses = setup_training_environment(config)
    optimizerG, optimizerD = create_optimizers(netG, netD, config)
    fixed_noise = create_fixed_noise(config, device)
    
    print("Starting Training Loop...")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        for i, data in enumerate(dataloader, 0):
            # Train Discriminator
            errD, fake, D_x, D_G_z1 = train_discriminator(
                netD, netG, data, criterion, optimizerD, config, device
            )
            
            # Train Generator
            errG, D_G_z2 = train_generator(
                netD, netG, fake, criterion, optimizerG, config, device
            )
            
            # Save losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
        # Save epoch images
        img_grid = save_epoch_images(netG, fixed_noise, epoch, config)
        img_list.append(img_grid)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{config["epochs"]} | '
              f'Time: {epoch_time:.2f}s | '
              f'Loss_D: {errD.item():.4f} | '
              f'Loss_G: {errG.item():.4f}')
    
    return G_losses, D_losses, img_list

def save_models(netG, netD, config):
    """Save the trained models."""
    torch.save(netG.state_dict(), config['generator_path'])
    torch.save(netD.state_dict(), config['discriminator_path'])
    
    print("Models saved successfully!")
    print(f"Generator weights saved to: {config['generator_path']}")
    print(f"Discriminator weights saved to: {config['discriminator_path']}")

def plot_losses(G_losses, D_losses):
    """Plot training losses."""
    if G_losses and D_losses:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("Loss history not found. Skipping loss plot.")

def visualize_results(netG, dataloader, config, device):
    """Visualize real vs fake images."""
    # Get real images
    real_batch = next(iter(dataloader))
    
    # Generate fake images
    with torch.no_grad():
        noise = torch.randn(16, config['latent_dim'], 1, 1, device=device)
        fake_images = netG(noise).detach().cpu()
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    plt.suptitle("DCGAN: Real vs. Fake Image Comparison", fontsize=16)
    
    # Real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    real_grid = vutils.make_grid(real_batch[0][:16], nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(real_grid, (1, 2, 0)))
    
    # Fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    fake_grid = vutils.make_grid(fake_images, nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
    
    plt.show()

def main():
    """Main function to orchestrate the training process."""
    # Setup
    device = get_device()
    dataloader = create_dataloader(CONFIG)
    netG, netD = initialize_models(CONFIG, device)
    
    # Training
    G_losses, D_losses, img_list = train_gan(netG, netD, dataloader, CONFIG, device)
    
    # Save models
    save_models(netG, netD, CONFIG)
    
    # Visualization
    plot_losses(G_losses, D_losses)
    visualize_results(netG, dataloader, CONFIG, device)
    
    return netG, netD, G_losses, D_losses, img_list

# Run the training if this script is executed directly
if __name__ == "__main__":
    netG, netD, G_losses, D_losses, img_list = main()