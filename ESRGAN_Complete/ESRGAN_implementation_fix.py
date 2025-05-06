import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from RRDBNet_arch import RRDBNet, RRDBNet_TinyFaces
import glob
from tqdm import tqdm
import random
import argparse
import cv2
from torch.nn import functional as F

# Configuration
class Config:
    def __init__(self):
        self.batch_size = 16
        self.lr = 2e-4
        self.num_epochs = 100
        self.save_interval = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale_factor = 4  # 4x upscaling
        self.hr_size = 256  # High-resolution patch size
        self.lr_size = self.hr_size // self.scale_factor  # Low-resolution patch size
        self.wider_face_path = "C:\\coding\\transformer\\ESRGAN_Complete\\wider_face"  # Update this to your Wider Face dataset path
        self.output_dir = "output"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.sample_dir = os.path.join(self.output_dir, "samples")

# Dataset preparation

# ---------- Fix 1: Face-centric dataset preparation ---------- 
class FaceDataset(Dataset):
    def __init__(self, root_dir, hr_size, scale, train=True):
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.valid_indices = []
        # Load face annotations
        folder = "WIDER_train" if train else "WIDER_val"
        self.image_paths = []
        self.face_annotations = {}
        
        # Parse Wider Face annotations
        annotation_file = os.path.join(
            root_dir, "wider_face_split",
            "wider_face_train_bbx_gt.txt" if train else "wider_face_val_bbx_gt.txt"
        )
        self.image_paths = []
        self.face_annotations = {}

        with open(annotation_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # Clean and skip empty lines

            i = 0
            while i < len(lines):
                img_path = lines[i]
    
                try:
                    num_faces = int(lines[i + 1])
                except (IndexError, ValueError):
                    raise ValueError(f"Line {i + 2}: Expected number of faces, got '{lines[i + 1] if i + 1 < len(lines) else 'EOF'}'")

                boxes = []
                for j in range(num_faces):
                    if i + 2 + j >= len(lines):
                        raise ValueError(f"Unexpected end of file while reading boxes for '{img_path}'")
        
                    parts = lines[i + 2 + j].split()
                    if len(parts) < 4:
                        raise ValueError(f"Line {i + 3 + j}: Invalid bounding box format: '{lines[i + 2 + j]}'")
        
                    try:
                        box = list(map(int, parts[:4]))
                    except ValueError:
                        raise ValueError(f"Line {i + 3 + j}: Could not convert bounding box values to integers: {parts[:4]}")
        
                    boxes.append(box)

                self.face_annotations[img_path] = boxes
                self.image_paths.append(os.path.join(root_dir, folder, "images", img_path))
                i += 2 + num_faces
        
        for idx, img_path in enumerate(self.image_paths):
            key = os.path.relpath(img_path, os.path.dirname(os.path.dirname(img_path)))
            if key in self.face_annotations:
                self.valid_indices.append(idx)
            
        print(f"Loaded {len(self.valid_indices)} valid images (with annotations)")

    def __len__(self):
        return len(self.valid_indices)

    def degrade_image(self, hr_image):
        """Fix 2: Realistic degradation pipeline"""
        # Convert to numpy for OpenCV operations
        hr_np = np.array(hr_image)
        
        # Add realistic blur
        hr_blur = cv2.GaussianBlur(hr_np, (5, 5), 0)
        
        # Downsample with nearest neighbor
        lr_np = cv2.resize(hr_blur, (self.lr_size, self.lr_size), 
                          interpolation=cv2.INTER_NEAREST)
        
        # Add noise
        noise = np.random.normal(0, 8, lr_np.shape).astype(np.uint8)
        lr_np = cv2.add(lr_np, noise)
        
        # JPEG compression simulation
        _, enc = cv2.imencode('.jpg', lr_np, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        lr_np = cv2.imdecode(enc, 1)
        
        return Image.fromarray(lr_np)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]  # Map to original index
        img_path = self.image_paths[real_idx]
        hr_image = Image.open(img_path).convert('RGB')
        key = os.path.normpath(os.path.relpath(img_path, os.path.dirname(os.path.dirname(img_path))))
        boxes = self.face_annotations.get(key)

        if boxes is None:
            raise KeyError(f"Annotation missing for image: {key}")

    
        # Initialize default box (full image coordinates)
        adjusted_box = [0, 0, self.hr_size, self.hr_size]  # Directly use HR size as fallback
    
        if len(boxes) > 0:
            # Select smallest face
            face_areas = [w*h for x,y,w,h in boxes]
            selected_idx = np.argmin(face_areas)
            x, y, w, h = boxes[selected_idx]
        
            # Calculate crop coordinates
            cx, cy = x + w//2, y + h//2
            size = max(w, h, self.hr_size//2)
            left = max(0, cx - size//2)
            top = max(0, cy - size//2)
            right = min(hr_image.width, cx + size//2)
            bottom = min(hr_image.height, cy + size//2)
        
            # Crop and calculate original dimensions BEFORE resizing
            cropped_hr = hr_image.crop((left, top, right, bottom))
            orig_crop_width = right - left
            orig_crop_height = bottom - top
        
            # Calculate box coordinates relative to crop
            new_x = x - left
            new_y = y - top
            new_w = w
            new_h = h
        
            # Resize cropped image to HR size
            hr_image = cropped_hr.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        
            # Calculate scaling factors
            width_ratio = self.hr_size / orig_crop_width
            height_ratio = self.hr_size / orig_crop_height
        
            # Adjust box coordinates for final HR image
            adjusted_box = [
                int(new_x * width_ratio),
                int(new_y * height_ratio),
                int(new_w * width_ratio),
                int(new_h * height_ratio)
            ]

        else:  # No faces case
            hr_image = hr_image.resize((self.hr_size, self.hr_size), Image.BICUBIC)
    
        # Degrade to create LR image
        lr_image = self.degrade_image(hr_image)
    
        return {
            'lr': self.transform(lr_image),
            'hr': self.transform(hr_image),
            'path': img_path,
            'boxes': [adjusted_box]  # Wrap in list for batch compatibility
        }

# Loss functions
class PerceptualLoss(nn.Module):
    def __init__(self, device):
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

# Trainer class
class ESRGANTrainer:
    def __init__(self, config):
        self.config = config
        
        # Create output directories
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.sample_dir, exist_ok=True)
        
        # Initialize networks
        self.generator = RRDBNet_TinyFaces(
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
        self.generator.eval()
        psnr_values = []
        ssim_values = []
        edge_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                lr_imgs = batch['lr'].to(self.config.device)
                hr_imgs = batch['hr'].to(self.config.device)
                sr_imgs = self.generator(lr_imgs)
                
                # Convert to [0,1] range
                hr = (hr_imgs + 1) / 2
                sr = (sr_imgs + 1) / 2
                
                # 1. Full-image PSNR
                mse = torch.mean((sr - hr) ** 2)
                psnr_values.append(10 * torch.log10(1 / mse).item())
                
                # 2. Face-region metrics using annotations
                if 'boxes' in batch:  # Add face boxes to dataset
                    for i in range(hr.shape[0]):
                        # Get face region coordinates
                        x1, y1, w, h = batch['boxes'][i]
                        x2, y2 = x1 + w, y1 + h
                        
                        # Extract face regions
                        hr_face = hr[i, :, y1:y2, x1:x2]
                        sr_face = sr[i, :, y1:y2, x1:x2]
                        
                        # Face-region SSIM
                        ssim = self.calculate_ssim(hr_face, sr_face)
                        ssim_values.append(ssim)
                        
                        # Edge preservation
                        edge_loss = self.calculate_edge_loss(hr_face, sr_face)
                        edge_losses.append(edge_loss)

        # Calculate averages
        metrics = {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values) if ssim_values else 0,
            'edge_loss': np.mean(edge_losses) if edge_losses else 0
        }
        
        print(f"Validation - PSNR: {metrics['psnr']:.2f} dB | "
              f"SSIM: {metrics['ssim']:.3f} | "
              f"Edge Loss: {metrics['edge_loss']:.4f}")
        return metrics['psnr']

    def calculate_ssim(self, hr, sr):
        # Implement SSIM calculation
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        return ssim(sr.unsqueeze(0), hr.unsqueeze(0)).item()

    def calculate_edge_loss(self, hr, sr):
        # Sobel edge detection
        sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32)
        
        hr_edges_x = F.conv2d(hr.unsqueeze(0), sobel_x)
        hr_edges_y = F.conv2d(hr.unsqueeze(0), sobel_y)
        hr_edges = torch.sqrt(hr_edges_x**2 + hr_edges_y**2)
        
        sr_edges_x = F.conv2d(sr.unsqueeze(0), sobel_x)
        sr_edges_y = F.conv2d(sr.unsqueeze(0), sobel_y)
        sr_edges = torch.sqrt(sr_edges_x**2 + sr_edges_y**2)
        
        return F.l1_loss(sr_edges, hr_edges).item()
    
    def save_model(self, epoch, is_best=False):
        """Save the model state"""
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
        """Save sample SR images"""
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

# Function to upscale all images in a directory
def upscale_directory(model_path, input_dir, output_dir, scale_factor=4, batch_size=4):
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
            output = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            
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