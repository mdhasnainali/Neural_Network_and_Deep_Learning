import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Configuration
CONFIG = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'workers': 4,
    'data_dir': './datasets/celebA',
    'image_size': 64,
    'channels': 3,
    'batch_size': 256,
    'latent_dim': 100,
    'epochs': 50,
    'learning_rate': 0.0002,
    'adam_beta_1': 0.5,
    'num_classes': 2,
    'label_embedding_dim': 50,
    'output_dir': 'cgan_longhair_images',
    'generator_path': 'cgan_longhair_generator.pth',
    'discriminator_path': 'cgan_longhair_discriminator.pth'
}

print(f"Using device: {CONFIG['device']}")

def create_celeba_dataset(config):
    """Create CelebA dataset with long hair labels."""
    img_dir_path = os.path.join(config['data_dir'], 'img_align_celeba')
    attr_file_path = os.path.join(config['data_dir'], 'list_attr_celeba.txt')
    
    # Custom dataset creation function
    def dataset_factory(img_dir, attr_file, transform=None):
        # Read attributes
        attr = pd.read_csv(attr_file, delim_whitespace=True, header=1)
        # Define long hair label (using Blond_Hair as proxy)
        long_hair_label = attr['Blond_Hair'].replace(-1, 0)
        labels = long_hair_label.values
        img_names = attr.index.values
        
        # Create dataset items
        dataset_items = []
        for idx in range(len(labels)):
            img_name = img_names[idx]
            img_path = os.path.join(img_dir, img_name)
            label = labels[idx]
            dataset_items.append((img_path, label))
        
        return dataset_items, labels
    
    transformations = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset_items, labels = dataset_factory(img_dir_path, attr_file_path, transformations)
    
    # Print dataset statistics
    long_hair_count = (labels == 1).sum()
    short_hair_count = (labels == 0).sum()
    print(f"Dataset loaded. Found {len(dataset_items)} images.")
    print(f"Long Hair (label 1): {long_hair_count}")
    print(f"Not Long Hair (label 0): {short_hair_count}")
    
    return dataset_items, transformations

def create_dataloader(dataset_items, transform, config):
    """Create dataloader using functional approach."""
    def collate_fn(batch):
        images = []
        labels = []
        
        for img_path, label in batch:
            try:
                image = Image.open(img_path).convert('RGB')
                if transform:
                    image = transform(image)
                images.append(image)
                labels.append(label)
            except Exception as e:
                continue
        
        if not images:
            return None, None
        
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)
    
    # Simple dataset wrapper
    class SimpleDataset(Dataset):
        def __init__(self, items):
            self.items = items
        
        def __len__(self):
            return len(self.items)
        
        def __getitem__(self, idx):
            return self.items[idx]
    
    dataset = SimpleDataset(dataset_items)
    return DataLoader(dataset, batch_size=config['batch_size'], 
                     shuffle=True, num_workers=config['workers'], 
                     collate_fn=collate_fn, drop_last=True)

def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def create_conditional_generator(config):
    """Create conditional generator using functional approach."""
    def forward_generator(noise, labels, label_embedding, main_layers):
        label_emb = label_embedding(labels).view(labels.size(0), config['label_embedding_dim'], 1, 1)
        gen_input = torch.cat((noise, label_emb), 1)
        return main_layers(gen_input)
    
    # Create components
    label_embedding = nn.Embedding(config['num_classes'], config['label_embedding_dim'])
    input_dim = config['latent_dim'] + config['label_embedding_dim']
    
    main_layers = nn.Sequential(
        nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
        nn.BatchNorm2d(512), nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128), nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(True),
        nn.ConvTranspose2d(64, config['channels'], 4, 2, 1, bias=False),
        nn.Tanh()
    )
    
    # Create a simple wrapper class
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_embedding = label_embedding
            self.main = main_layers
            
        def forward(self, noise, labels):
            return forward_generator(noise, labels, self.label_embedding, self.main)
    
    return Generator()

def create_conditional_discriminator(config):
    """Create conditional discriminator using functional approach."""
    def forward_discriminator(image, labels, image_path, label_path, classifier):
        # Process image and label separately
        image_features = image_path(image)
        label_features = label_path(labels).view(-1, 512, 4, 4)
        
        # Combine and classify
        combined_features = torch.cat([image_features, label_features], dim=1)
        return classifier(combined_features)
    
    # Create components
    image_path = nn.Sequential(
        nn.Conv2d(config['channels'], 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
    )
    
    label_path = nn.Sequential(
        nn.Embedding(config['num_classes'], 50),
        nn.Linear(50, 512 * 4 * 4)
    )
    
    classifier = nn.Sequential(
        nn.Conv2d(512 * 2, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
    
    # Create a simple wrapper class
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_path = image_path
            self.label_path = label_path
            self.classifier = classifier
            
        def forward(self, image, labels):
            return forward_discriminator(image, labels, self.image_path, 
                                       self.label_path, self.classifier)
    
    return Discriminator()

def initialize_models(config):
    """Initialize generator and discriminator."""
    netG = create_conditional_generator(config).to(config['device'])
    netD = create_conditional_discriminator(config).to(config['device'])
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print("Generator and Discriminator models created successfully.")
    return netG, netD

def train_discriminator_step(netD, real_images, real_labels, netG, criterion, config):
    """Single discriminator training step."""
    b_size = real_images.size(0)
    device = config['device']
    
    netD.zero_grad()
    
    # Train with real images
    label_true = torch.full((b_size,), 1., dtype=torch.float, device=device)
    output = netD(real_images, real_labels).view(-1)
    errD_real = criterion(output, label_true)
    errD_real.backward()
    
    # Train with fake images
    noise = torch.randn(b_size, config['latent_dim'], 1, 1, device=device)
    fake_labels = torch.randint(0, config['num_classes'], (b_size,), device=device).long()
    fake_images = netG(noise, fake_labels)
    
    label_false = torch.full((b_size,), 0., dtype=torch.float, device=device)
    output = netD(fake_images.detach(), fake_labels).view(-1)
    errD_fake = criterion(output, label_false)
    errD_fake.backward()
    
    errD = errD_real + errD_fake
    return errD, fake_images, fake_labels

def train_generator_step(netG, netD, fake_images, fake_labels, criterion, config):
    """Single generator training step."""
    netG.zero_grad()
    
    b_size = fake_images.size(0)
    label_true = torch.full((b_size,), 1., dtype=torch.float, device=config['device'])
    
    output = netD(fake_images, fake_labels).view(-1)
    errG = criterion(output, label_true)
    errG.backward()
    
    return errG

def save_epoch_images(netG, epoch, config):
    """Save generated images for current epoch."""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    with torch.no_grad():
        fixed_noise = torch.randn(16, config['latent_dim'], 1, 1, device=config['device'])
        labels_long = torch.full((16,), 1, dtype=torch.long, device=config['device'])
        fake_long_hair_grid = netG(fixed_noise, labels_long).detach().cpu()
        vutils.save_image(fake_long_hair_grid, 
                         f"{config['output_dir']}/epoch_{epoch+1}.png", 
                         normalize=True)

def train_cgan(netG, netD, dataloader, config):
    """Main training loop for conditional GAN."""
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), 
                           lr=config['learning_rate'], 
                           betas=(config['adam_beta_1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), 
                           lr=config['learning_rate'], 
                           betas=(config['adam_beta_1'], 0.999))
    
    G_losses = []
    D_losses = []
    
    print("Starting Conditional GAN Training...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        for i, (real_images, real_labels) in enumerate(dataloader):
            if real_images is None:  # Skip failed batches
                continue
                
            real_images = real_images.to(config['device'])
            real_labels = real_labels.to(config['device'])
            
            # Train Discriminator
            errD, fake_images, fake_labels = train_discriminator_step(
                netD, real_images, real_labels, netG, criterion, config)
            optimizerD.step()
            
            # Train Generator
            errG = train_generator_step(netG, netD, fake_images, fake_labels, 
                                     criterion, config)
            optimizerG.step()
            
            # Save losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
        # Save epoch images
        save_epoch_images(netG, epoch, config)
        
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{config["epochs"]}] | '
              f'Loss_D: {errD.item():.4f} | '
              f'Loss_G: {errG.item():.4f} | '
              f'Time: {epoch_time:.2f}s')
    
    return G_losses, D_losses

def save_models(netG, netD, config):
    """Save trained models."""
    torch.save(netG.state_dict(), config['generator_path'])
    torch.save(netD.state_dict(), config['discriminator_path'])
    print("\nTraining complete. Models saved.")

def plot_losses(G_losses, D_losses):
    """Plot training losses."""
    if G_losses and D_losses:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="Generator")
        plt.plot(D_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("Loss history not found. Skipping loss plot.")

def find_real_long_hair_images(dataloader, num_needed=16):
    """Find real images with long hair label."""
    real_long_hair_images = []
    
    for images, labels in dataloader:
        if images is None:
            continue
        long_hair_in_batch = images[labels == 1]
        if len(long_hair_in_batch) > 0:
            real_long_hair_images.append(long_hair_in_batch)
        
        if sum(len(b) for b in real_long_hair_images) >= num_needed:
            break
    
    if real_long_hair_images:
        return torch.cat(real_long_hair_images)[:num_needed]
    return None

def generate_fake_long_hair_images(netG, config, num_images=16):
    """Generate fake images with long hair label."""
    with torch.no_grad():
        noise = torch.randn(num_images, config['latent_dim'], 1, 1, device=config['device'])
        long_hair_labels = torch.full((num_images,), 1, dtype=torch.long, device=config['device'])
        return netG(noise, long_hair_labels).detach().cpu()

def visualize_results(netG, dataloader, config):
    """Visualize real vs fake long hair images."""
    print("Searching for 16 real 'long hair' images from the dataset...")
    real_long_hair_tensor = find_real_long_hair_images(dataloader)
    
    print("Generating 16 fake 'long hair' images...")
    fake_long_hair_images = generate_fake_long_hair_images(netG, config)
    
    if real_long_hair_tensor is None:
        print("Could not find enough real long hair images.")
        return
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    plt.suptitle("cGAN: Real vs. Fake 'Long Hair' Comparison", fontsize=16)
    
    # Real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images (Long Hair)")
    real_grid = vutils.make_grid(real_long_hair_tensor, nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(real_grid, (1, 2, 0)))
    
    # Fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images (Long Hair)")
    fake_grid = vutils.make_grid(fake_long_hair_images, nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
    
    plt.show()

def main():
    """Main execution function."""
    # Setup dataset and dataloader
    dataset_items, transform = create_celeba_dataset(CONFIG)
    dataloader = create_dataloader(dataset_items, transform, CONFIG)
    
    # Initialize models
    netG, netD = initialize_models(CONFIG)
    
    # Train the model
    G_losses, D_losses = train_cgan(netG, netD, dataloader, CONFIG)
    
    # Save models
    save_models(netG, netD, CONFIG)
    
    # Visualize results
    plot_losses(G_losses, D_losses)
    
    # Set generator to eval mode for visualization
    netG.eval()
    visualize_results(netG, dataloader, CONFIG)
    
    return netG, netD, G_losses, D_losses

# Run training if executed directly
if __name__ == '__main__':
    netG, netD, G_losses, D_losses = main()