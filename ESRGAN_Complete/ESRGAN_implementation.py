# Standard library
import argparse
import glob
import os
import random

# Third-party
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Local
from RRDBNet_arch import RRDBNet

# Configuration
class Config:
    """Container for training/inference hyperparameters and paths."""

    def __init__(self):
        """Initialize default configuration values (no inputs, no outputs)."""
        self.batch_size = 16
        self.lr = 2e-4
        self.num_epochs = 100
        self.save_interval = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale_factor = 4  # 4x upscaling
        self.hr_size = 128  # High-resolution patch size
        self.lr_size = self.hr_size // self.scale_factor  # Low-resolution patch size
        self.wider_face_path = "C:\\coding\\transformer\\ESRGAN_Complete\\wider_face"  # Update this to your Wider Face dataset path
        self.output_dir = "output"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.sample_dir = os.path.join(self.output_dir, "samples")

# =============================================================================
# DATASET LOADING
# =============================================================================

# Dataset preparation
class FaceDataset(Dataset):
    """WiderFace dataset that yields (LR, HR) image pairs via random crops."""

    def __init__(self, root_dir, hr_size, scale, train=True):
        """Index every image in the chosen WiderFace split.

        Args:
            root_dir (string): Directory with the Wider Face dataset
            hr_size (int): Size of high-resolution output patches
            scale (int): Downsampling scale factor
            train (bool): If True, create dataset from training folder, else from validation folder
        """
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Get image paths
        folder = "WIDER_train" if train else "WIDER_val"
        image_dir = os.path.join(root_dir, folder, "images")
        self.image_paths = []
        
        # Recursively find all JPG images
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} images in {folder} set")

    def __len__(self):
        """Return the number of indexed images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return one (LR, HR) image pair from a random crop.

        Args:
            idx (int): Index into ``self.image_paths``.

        Returns:
            dict: Keys ``lr``/``hr`` (tensors) and ``path`` (str).
        """
        img_path = self.image_paths[idx]
        
        # Read the image
        hr_image = Image.open(img_path).convert('RGB')
        
        # Random crop
        if hr_image.width < self.hr_size or hr_image.height < self.hr_size:
            # If image is too small, resize it to at least hr_size
            ratio = max(self.hr_size / hr_image.width, self.hr_size / hr_image.height)
            new_width = int(hr_image.width * ratio)
            new_height = int(hr_image.height * ratio)
            hr_image = hr_image.resize((new_width, new_height), Image.BICUBIC)
        
        # Random crop
        left = random.randint(0, hr_image.width - self.hr_size)
        top = random.randint(0, hr_image.height - self.hr_size)
        hr_image = hr_image.crop((left, top, left + self.hr_size, top + self.hr_size))
        
        # ---------------------------------------------------------------------
        # LOW-RESOLUTION IMAGE GENERATION
        # ---------------------------------------------------------------------
        # Create LR image through downsampling
        lr_image = hr_image.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        
        # Convert to tensors
        hr_tensor = self.transform(hr_image)
        lr_tensor = self.transform(lr_image)
        
        return {'lr': lr_tensor, 'hr': hr_tensor, 'path': img_path}

# =============================================================================
# MODEL DEFINITION (loss functions)
# =============================================================================

# Loss functions
class PerceptualLoss(nn.Module):
    """VGG19 feature-space MSE loss between SR and HR images."""

    def __init__(self, device):
        """Load a frozen VGG19 feature extractor on the given device.

        Args:
            device (torch.device): Device to place the VGG features on.
        """
        super(PerceptualLoss, self).__init__()
        # Use VGG19 features for perceptual loss - load directly from torchvision
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[:35]).to(device)
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.device = device
    
    def forward(self, sr, hr):
        """Compute perceptual MSE loss between SR and HR feature maps.

        Args:
            sr (torch.Tensor): Super-resolved image batch in [-1, 1].
            hr (torch.Tensor): Ground-truth HR image batch in [-1, 1].

        Returns:
            torch.Tensor: Scalar perceptual loss.
        """
        # Denormalize from [-1,1] to [0,1]
        sr = (sr + 1) / 2
        hr = (hr + 1) / 2
        
        # Convert to range expected by VGG [0,1] -> [0,255] and normalize with ImageNet stats
        sr = sr * 255
        hr = hr * 255
        
        # ImageNet mean and std
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        sr = (sr / 255 - mean) / std
        hr = (hr / 255 - mean) / std
        
        sr_features = self.vgg_features(sr)
        hr_features = self.vgg_features(hr)
        
        return self.mse_loss(sr_features, hr_features)

# =============================================================================
# TRAINING LOOP
# =============================================================================

# Trainer class
class ESRGANTrainer:
    """Builds the model, datasets and runs the ESRGAN training loop."""

    def __init__(self, config):
        """Initialize the model, optimizer, losses and data loaders.

        Args:
            config (Config): Training configuration object.
        """
        self.config = config
        
        # Create output directories
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.sample_dir, exist_ok=True)
        
        # Initialize networks
        self.generator = RRDBNet(
            in_nc=3,         # RGB input
            out_nc=3,        # RGB output
            nf=64,           # Number of features
            nb=23,           # Number of RRDB blocks
            gc=32            # Growth channel
        ).to(config.device)
        
        # Initialize optimizer
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=config.lr, betas=(0.9, 0.999))
        
        # Define loss functions
        self.l1_loss = nn.L1Loss().to(config.device)
        self.perceptual_loss = PerceptualLoss(config.device)
        
        # Create datasets
        self.train_dataset = FaceDataset(
            root_dir=config.wider_face_path,
            hr_size=config.hr_size,
            scale=config.scale_factor,
            train=True
        )
        
        self.val_dataset = FaceDataset(
            root_dir=config.wider_face_path,
            hr_size=config.hr_size,
            scale=config.scale_factor,
            train=False
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
    
    def train(self):
        """Run the full training loop, validating and saving periodically.

        No inputs. No return value; checkpoints and samples are written to disk.
        """
        best_psnr = 0
        
        for epoch in range(self.config.num_epochs):
            self.generator.train()
            
            pbar = tqdm(self.train_loader)
            for batch in pbar:
                # Get data
                lr_imgs = batch['lr'].to(self.config.device)
                hr_imgs = batch['hr'].to(self.config.device)
                
                # Forward pass
                self.optimizer_G.zero_grad()
                sr_imgs = self.generator(lr_imgs)
                
                # Calculate losses
                pixel_loss = self.l1_loss(sr_imgs, hr_imgs)
                p_loss = self.perceptual_loss(sr_imgs, hr_imgs)
                
                # Total loss (you can adjust weights)
                total_loss = pixel_loss + 0.1 * p_loss
                
                # Backward pass and optimize
                total_loss.backward()
                self.optimizer_G.step()
                
                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {total_loss.item():.4f}")
            
            # Validation
            if (epoch + 1) % self.config.save_interval == 0:
                avg_psnr = self.validate()
                print(f"Epoch {epoch+1} - Validation PSNR: {avg_psnr:.4f}")
                
                # Save model
                self.save_model(epoch + 1)
                
                # Save sample images
                self.save_samples(epoch + 1)
                
                # Save best model
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    self.save_model(epoch + 1, is_best=True)
    
    def validate(self):
        """Evaluate the generator on the validation set.

        Returns:
            float: Mean PSNR (dB) over the validation images.
        """
        self.generator.eval()
        psnr_values = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                lr_imgs = batch['lr'].to(self.config.device)
                hr_imgs = batch['hr'].to(self.config.device)
                
                sr_imgs = self.generator(lr_imgs)
                
                # Calculate PSNR (from [-1,1] range to [0,1] range)
                sr = (sr_imgs + 1) / 2
                hr = (hr_imgs + 1) / 2
                
                mse = torch.mean((sr - hr) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                psnr_values.append(psnr.item())
        
        return sum(psnr_values) / len(psnr_values)
    
    def save_model(self, epoch, is_best=False):
        """Save the generator and optimizer state to disk.

        Args:
            epoch (int): Current epoch number used in the file name.
            is_best (bool): If True, save under ``best_model.pth``.
        """
        if is_best:
            path = os.path.join(self.config.models_dir, f"best_model.pth")
        else:
            path = os.path.join(self.config.models_dir, f"model_epoch_{epoch}.pth")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer_G.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def save_samples(self, epoch):
        """Generate and save LR/HR/SR sample images for visual inspection.

        Args:
            epoch (int): Current epoch number used in the file names.
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Take first batch from validation set
            batch = next(iter(self.val_loader))
            lr_imgs = batch['lr'].to(self.config.device)
            hr_imgs = batch['hr'].to(self.config.device)
            
            # Generate SR images
            sr_imgs = self.generator(lr_imgs)
            
            # Convert tensors to numpy arrays (from [-1,1] to [0,255])
            lr_img = (lr_imgs[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
            hr_img = (hr_imgs[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
            sr_img = (sr_imgs[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
            
            # Clip values to [0, 255]
            lr_img = np.clip(lr_img, 0, 255).astype(np.uint8)
            hr_img = np.clip(hr_img, 0, 255).astype(np.uint8)
            sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)
            
            # Save images
            cv2.imwrite(os.path.join(self.config.sample_dir, f"epoch_{epoch}_lr.png"), cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.config.sample_dir, f"epoch_{epoch}_hr.png"), cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.config.sample_dir, f"epoch_{epoch}_sr.png"), cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

# =============================================================================
# INFERENCE / UPSCALING
# =============================================================================

# Function to upscale all images in a directory
def upscale_directory(model_path, input_dir, output_dir, scale_factor=4, batch_size=4):
    """Run the trained generator over every image in a directory.

    Args:
        model_path (str): Path to a saved checkpoint ``.pth`` file.
        input_dir (str): Folder of images to upscale (recursive).
        output_dir (str): Folder where upscaled images will be written.
        scale_factor (int): Final upscaling factor applied via resize.
        batch_size (int): Number of images per inference batch.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Get all image paths
    image_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Upscaling images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        original_sizes = []
        
        # Prepare batch
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            original_sizes.append((img.width, img.height))
            tensor = transform(img).unsqueeze(0)
            batch_imgs.append(tensor)
        
        # Concatenate tensors
        input_batch = torch.cat(batch_imgs, dim=0).to(device)
        
        # Process batch
        with torch.no_grad():
            output_batch = model(input_batch)
        
        # Save output images
        for j, path in enumerate(batch_paths):
            # Get output image
            output = output_batch[j]
            
            # Convert to numpy and rescale to [0, 255]
            output = output.permute(1, 2, 0).cpu().numpy()
            output = (output + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Resize to target size (4x original)
            target_width = original_sizes[j][0] * scale_factor
            target_height = original_sizes[j][1] * scale_factor
            output = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Construct output path
            rel_path = os.path.relpath(path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            cv2.imwrite(output_path, output)
            print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESRGAN for Wider Face Dataset")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'upscale'],
                        help='train: train ESRGAN model, upscale: upscale images')
    parser.add_argument('--model_path', type=str, default='output/models/best_model.pth',
                        help='Path to saved model (for upscale mode)')
    parser.add_argument('--input_dir', type=str, default='input',
                        help='Input directory containing images to upscale (for upscale mode)')
    parser.add_argument('--output_dir', type=str, default='upscaled',
                        help='Output directory for upscaled images (for upscale mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = Config()
        trainer = ESRGANTrainer(config)
        trainer.train()
    else:
        upscale_directory(
            model_path=args.model_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )