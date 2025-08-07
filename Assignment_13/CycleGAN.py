import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Configuration
CONFIG = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'workers': 4,
    'data_root': '/media/cseru/c7affc2f-7569-4a5f-af53-c40816938344/Shakil/Datasets/cycleGAN',
    'image_size': 128,
    'channels': 3,
    'batch_size': 256,
    'epochs': 50,
    'learning_rate': 0.0002,
    'adam_beta_1': 0.5,
    'decay_start_epoch': 10,
    'lambda_cycle': 10.0,
    'lambda_identity': 5.0,
    'output_dir_samples': 'cyclegan_samples',
    'output_dir_models': 'cyclegan_models',
    'generator_a_to_b_path': 'cyclegan_models/G_AtoB.pth'
}

print(f"Using device: {CONFIG['device']}")

def get_image_files(data_root):
    """Get image file paths from trainA and trainB directories."""
    files_A = sorted(glob.glob(os.path.join(data_root, 'trainA') + '/*.*'))
    files_B = sorted(glob.glob(os.path.join(data_root, 'trainB') + '/*.*'))
    print(f"Found {len(files_A)} images in trainA and {len(files_B)} images in trainB.")
    return files_A, files_B

def load_image_pair(files_A, files_B, index, transform, unaligned=True):
    """Load and transform a pair of images from domains A and B."""
    # Load image from Domain A
    image_A = Image.open(files_A[index % len(files_A)]).convert('RGB')
    
    # Load image from Domain B
    if unaligned:
        index_B = random.randint(0, len(files_B) - 1)
    else:
        index_B = index % len(files_B)
    image_B = Image.open(files_B[index_B]).convert('RGB')
    
    # Apply transformations
    if transform:
        item_A = transform(image_A)
        item_B = transform(image_B)
    
    return {'A': item_A, 'B': item_B}

def create_transforms(config):
    """Create image transformations."""
    return transforms.Compose([
        transforms.Resize(int(config['image_size'] * 1.12), Image.BICUBIC),
        transforms.RandomCrop(config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def create_dataloader(config):
    """Create dataloader using functional approach."""
    files_A, files_B = get_image_files(config['data_root'])
    transform = create_transforms(config)
    
    def collate_fn(indices):
        batch = []
        for idx in indices:
            try:
                pair = load_image_pair(files_A, files_B, idx, transform)
                batch.append(pair)
            except Exception as e:
                continue
        
        if not batch:
            return None
        
        batch_A = torch.stack([item['A'] for item in batch])
        batch_B = torch.stack([item['B'] for item in batch])
        return {'A': batch_A, 'B': batch_B}
    
    # Simple dataset wrapper
    class SimpleDataset:
        def __init__(self, length):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return idx
    
    dataset = SimpleDataset(max(len(files_A), len(files_B)))
    return DataLoader(dataset, batch_size=config['batch_size'], 
                     shuffle=True, num_workers=config['workers'], 
                     collate_fn=collate_fn, drop_last=True)

def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def create_residual_block(in_features):
    """Create a residual block."""
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.InstanceNorm2d(in_features)
    )

def create_generator_resnet(input_shape, num_residual_blocks=9):
    """Create ResNet-based generator."""
    channels = input_shape[0]
    
    # Minimal wrapper class for residual connection
    class ResidualBlock(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
        def forward(self, x):
            return x + self.block(x)
    
    # Initial convolution block
    out_features = 64
    model = [
        nn.ReflectionPad2d(channels),
        nn.Conv2d(channels, out_features, 7),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True)
    ]
    
    in_features = out_features
    
    # Downsampling
    for _ in range(2):
        out_features *= 2
        model += [
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
    
    # Residual blocks
    for _ in range(num_residual_blocks):
        model.append(ResidualBlock(create_residual_block(out_features)))
    
    # Upsampling
    for _ in range(2):
        out_features //= 2
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
    
    # Output layer
    model += [
        nn.ReflectionPad2d(channels),
        nn.Conv2d(out_features, channels, 7),
        nn.Tanh()
    ]
    
    return nn.Sequential(*model)

def create_discriminator_block(in_filters, out_filters, normalize=True):
    """Create a discriminator block."""
    layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

def create_discriminator(input_shape):
    """Create PatchGAN discriminator."""
    channels, height, width = input_shape
    
    model = nn.Sequential(
        *create_discriminator_block(channels, 64, normalize=False),
        *create_discriminator_block(64, 128),
        *create_discriminator_block(128, 256),
        *create_discriminator_block(256, 512),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(512, 1, 4, padding=1)
    )
    
    # Add output shape as attribute
    model.output_shape = (1, height // 2**4, width // 2**4)
    return model

def initialize_models(config):
    """Initialize all CycleGAN models."""
    input_shape = (config['channels'], config['image_size'], config['image_size'])
    
    G_AtoB = create_generator_resnet(input_shape).to(config['device'])
    G_BtoA = create_generator_resnet(input_shape).to(config['device'])
    D_A = create_discriminator(input_shape).to(config['device'])
    D_B = create_discriminator(input_shape).to(config['device'])
    
    # Apply weight initialization
    for model in [G_AtoB, G_BtoA, D_A, D_B]:
        model.apply(weights_init)
    
    print("CycleGAN models created successfully.")
    return G_AtoB, G_BtoA, D_A, D_B

def create_loss_functions(config):
    """Create loss functions."""
    criterion_GAN = nn.MSELoss().to(config['device'])
    criterion_cycle = nn.L1Loss().to(config['device'])
    criterion_identity = nn.L1Loss().to(config['device'])
    return criterion_GAN, criterion_cycle, criterion_identity

def create_optimizers(G_AtoB, G_BtoA, D_A, D_B, config):
    """Create optimizers for all models."""
    optimizer_G = optim.Adam(
        itertools.chain(G_AtoB.parameters(), G_BtoA.parameters()),
        lr=config['learning_rate'], 
        betas=(config['adam_beta_1'], 0.999)
    )
    optimizer_D_A = optim.Adam(
        D_A.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['adam_beta_1'], 0.999)
    )
    optimizer_D_B = optim.Adam(
        D_B.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['adam_beta_1'], 0.999)
    )
    return optimizer_G, optimizer_D_A, optimizer_D_B

def create_lr_schedulers(optimizers, config):
    """Create learning rate schedulers."""
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - config['decay_start_epoch']) / (config['epochs'] - config['decay_start_epoch'])
    
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)
    
    return lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B

def create_replay_buffer(max_size=50):
    """Create replay buffer using functional approach."""
    buffer_data = []
    
    def push_and_pop(data):
        nonlocal buffer_data
        to_return = []
        
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(buffer_data) < max_size:
                buffer_data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, max_size - 1)
                    to_return.append(buffer_data[i].clone())
                    buffer_data[i] = element
                else:
                    to_return.append(element)
        
        return torch.cat(to_return) if to_return else data
    
    return push_and_pop

def train_generators(G_AtoB, G_BtoA, D_A, D_B, real_A, real_B, target_real, 
                    criterion_GAN, criterion_cycle, criterion_identity, config):
    """Train generator networks for one step."""
    G_AtoB.train()
    G_BtoA.train()
    
    # Identity loss
    loss_id_A = criterion_identity(G_BtoA(real_A), real_A)
    loss_id_B = criterion_identity(G_AtoB(real_B), real_B)
    loss_identity = (loss_id_A + loss_id_B) / 2

    # GAN loss
    fake_B = G_AtoB(real_A)
    loss_GAN_AtoB = criterion_GAN(D_B(fake_B), target_real)
    fake_A = G_BtoA(real_B)
    loss_GAN_BtoA = criterion_GAN(D_A(fake_A), target_real)
    loss_GAN = (loss_GAN_AtoB + loss_GAN_BtoA) / 2

    # Cycle loss
    reconstructed_A = G_BtoA(fake_B)
    loss_cycle_A = criterion_cycle(reconstructed_A, real_A)
    reconstructed_B = G_AtoB(fake_A)
    loss_cycle_B = criterion_cycle(reconstructed_B, real_B)
    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

    # Total generator loss
    loss_G = loss_GAN + config['lambda_cycle'] * loss_cycle + config['lambda_identity'] * loss_identity
    
    return loss_G, fake_A, fake_B

def train_discriminator(D, real_imgs, fake_imgs, target_real, target_fake, 
                       criterion_GAN, fake_buffer):
    """Train discriminator for one step."""
    # Real loss
    loss_real = criterion_GAN(D(real_imgs), target_real)
    
    # Fake loss with replay buffer
    fake_imgs_buffered = fake_buffer(fake_imgs)
    loss_fake = criterion_GAN(D(fake_imgs_buffered.detach()), target_fake)
    
    # Total loss
    loss_D = (loss_real + loss_fake) / 2
    return loss_D

def save_sample_images(G_AtoB, G_BtoA, real_A, real_B, epoch, config):
    """Save sample generated images."""
    sample_real_A = real_A[0].unsqueeze(0)
    sample_fake_B = G_AtoB(sample_real_A).detach()
    sample_real_B = real_B[0].unsqueeze(0)
    sample_fake_A = G_BtoA(sample_real_B).detach()
    
    image_grid = torch.cat((sample_real_A, sample_fake_B, sample_real_B, sample_fake_A), 0)
    vutils.save_image(image_grid, 
                     f"{config['output_dir_samples']}/epoch_{epoch+1}.png", 
                     nrow=4, normalize=True)

def save_models(G_AtoB, G_BtoA, D_A, D_B, config):
    """Save model checkpoints."""
    torch.save(G_AtoB.state_dict(), f"{config['output_dir_models']}/G_AtoB.pth")
    torch.save(G_BtoA.state_dict(), f"{config['output_dir_models']}/G_BtoA.pth")
    torch.save(D_A.state_dict(), f"{config['output_dir_models']}/D_A.pth")
    torch.save(D_B.state_dict(), f"{config['output_dir_models']}/D_B.pth")

def train_cyclegan(config):
    """Main training loop for CycleGAN."""
    # Create directories
    os.makedirs(config['output_dir_samples'], exist_ok=True)
    os.makedirs(config['output_dir_models'], exist_ok=True)
    
    # Initialize components
    dataloader = create_dataloader(config)
    G_AtoB, G_BtoA, D_A, D_B = initialize_models(config)
    criterion_GAN, criterion_cycle, criterion_identity = create_loss_functions(config)
    
    optimizers = create_optimizers(G_AtoB, G_BtoA, D_A, D_B, config)
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    
    schedulers = create_lr_schedulers(optimizers, config)
    lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = schedulers
    
    # Create replay buffers
    fake_A_buffer = create_replay_buffer()
    fake_B_buffer = create_replay_buffer()
    
    print("Starting CycleGAN Training...")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
                
            real_A = batch['A'].to(config['device'])
            real_B = batch['B'].to(config['device'])
            
            # Create target tensors
            target_real = torch.ones((real_A.size(0), *D_A.output_shape), 
                                   requires_grad=False).to(config['device'])
            target_fake = torch.zeros((real_A.size(0), *D_A.output_shape), 
                                    requires_grad=False).to(config['device'])
            
            # Train Generators
            optimizer_G.zero_grad()
            loss_G, fake_A, fake_B = train_generators(
                G_AtoB, G_BtoA, D_A, D_B, real_A, real_B, target_real,
                criterion_GAN, criterion_cycle, criterion_identity, config
            )
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_D_A = train_discriminator(
                D_A, real_A, fake_A, target_real, target_fake, 
                criterion_GAN, fake_A_buffer
            )
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_D_B = train_discriminator(
                D_B, real_B, fake_B, target_real, target_fake, 
                criterion_GAN, fake_B_buffer
            )
            loss_D_B.backward()
            optimizer_D_B.step()
        
        # End of epoch
        epoch_time = time.time() - start_time
        loss_D = (loss_D_A + loss_D_B) / 2
        print(f"Epoch [{epoch+1}/{config['epochs']}] | "
              f"Loss_D: {loss_D.item():.4f} | "
              f"Loss_G: {loss_G.item():.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        # Save sample images and models
        save_sample_images(G_AtoB, G_BtoA, real_A, real_B, epoch, config)
        save_models(G_AtoB, G_BtoA, D_A, D_B, config)
    
    return G_AtoB, G_BtoA, D_A, D_B

def load_generator_for_inference(config):
    """Load trained generator for inference."""
    input_shape = (config['channels'], config['image_size'], config['image_size'])
    G_AtoB = create_generator_resnet(input_shape).to(config['device'])
    
    try:
        G_AtoB.load_state_dict(torch.load(config['generator_a_to_b_path']))
        print(f"Successfully loaded weights from '{config['generator_a_to_b_path']}'")
        G_AtoB.eval()
        return G_AtoB
    except FileNotFoundError:
        print(f"Error: Model file not found at '{config['generator_a_to_b_path']}'")
        return None

def collect_real_images(dataloader, num_needed=16):
    """Collect real images from dataloader."""
    real_photos_list = []
    
    for batch in dataloader:
        if batch is None:
            continue
        real_photos_list.append(batch['A'])
        if sum(b.size(0) for b in real_photos_list) >= num_needed:
            break
    
    if real_photos_list:
        return torch.cat(real_photos_list, 0)[:num_needed]
    return None

def visualize_translation_results(G_AtoB, dataloader, config):
    """Visualize photo-to-painting translation results."""
    print("Fetching 16 real face photos from the dataset...")
    real_photos = collect_real_images(dataloader, 16)
    
    if real_photos is None:
        print("Could not collect real images.")
        return
    
    real_photos = real_photos.to(config['device'])
    
    print("Translating photos to the painted domain...")
    with torch.no_grad():
        fake_paintings = G_AtoB(real_photos).detach().cpu()
    
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.suptitle("CycleGAN: Photo-to-Painting Translation", fontsize=16)
    
    # Original photos
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Original Photos (Domain A)")
    real_grid = vutils.make_grid(real_photos.cpu(), nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(real_grid, (1, 2, 0)))
    
    # Generated paintings
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Paintings (Domain B)")
    fake_grid = vutils.make_grid(fake_paintings, nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
    
    plt.show()

def main():
    """Main execution function."""
    # Training
    print("Training CycleGAN...")
    G_AtoB, G_BtoA, D_A, D_B = train_cyclegan(CONFIG)
    
    # Inference and visualization
    print("\nLoading generator for inference...")
    G_AtoB_inference = load_generator_for_inference(CONFIG)
    
    if G_AtoB_inference is not None:
        dataloader = create_dataloader(CONFIG)
        visualize_translation_results(G_AtoB_inference, dataloader, CONFIG)
    
    return G_AtoB, G_BtoA, D_A, D_B

# Run if executed directly
if __name__ == '__main__':
    models = main()