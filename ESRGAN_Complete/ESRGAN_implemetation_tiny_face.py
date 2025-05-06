import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from RRDBNet_arch import RRDBNet
import glob
import json
from tqdm import tqdm
import random
import argparse
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import functools

# Configuration
class Config:
    def __init__(self):
        self.batch_size = 16
        self.lr = 2e-4
        self.num_epochs = 100
        self.save_interval = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale_factor = 4  # 4x upscaling
        self.hr_size = 128  # High-resolution patch size
        self.lr_size = self.hr_size // self.scale_factor  # Low-resolution patch size
        self.wider_face_path = "path/to/wider_face"  # Update this to your Wider Face dataset path
        self.output_dir = "output"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.sample_dir = os.path.join(self.output_dir, "samples")
        
        # Face detection parameters
        self.min_face_size = 10  # Minimum face size to include in training
        self.max_face_size = 32  # Target small faces for upscaling training
        
        # Realistic degradation parameters
        self.add_blur = True
        self.add_noise = True
        self.add_compression = True
        self.blur_kernel_size = 3
        self.noise_level = 5
        self.jpeg_quality = 75

# Enhanced network with more blocks for tiny face recovery
class RRDBNet_TinyFaces(RRDBNet):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        # Call parent constructor with more blocks and features
        super(RRDBNet_TinyFaces, self).__init__(in_nc, out_nc, nf, nb, gc)
        
        # Override with additional upscaling capability for finer details
        self.upconv3 = nn.Conv2d(nf, nf//2, 3, 1, 1, bias=True)
        self.HRconv2 = nn.Conv2d(nf//2, nf//2, 3, 1, 1, bias=True)
        
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        # Extra refinement for facial details
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

# Helper for annotation parsing
def parse_wider_face_annotations(annotation_file):
    """Parse WiderFace annotation file and return a dictionary of image paths to face boxes"""
    annotations = {}
    current_img = None
    face_count = 0
    
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith('.jpg') or line.endswith('.png'):
            current_img = line
            annotations[current_img] = []
            face_count = int(lines[i+1])
            i += 2
            for j in range(face_count):
                if i+j < len(lines):
                    values = lines[i+j].strip().split()
                    # Format: [x, y, w, h, blur, expression, illumination, invalid, occlusion, pose]
                    if len(values) >= 4:  # At least x,y,w,h should be present
                        x, y, w, h = map(float, values[:4])
                        # Only store valid faces (not marked as invalid)
                        if len(values) <= 7 or int(values[7]) == 0:
                            annotations[current_img].append([x, y, w, h])
            i += face_count
        else:
            i += 1
    
    return annotations

# Dataset with face-aware sampling
class FaceDataset(Dataset):
    def __init__(self, root_dir, hr_size, scale, train=True, config=None):
        """
        Args:
            root_dir (string): Directory with the Wider Face dataset
            hr_size (int): Size of high-resolution output patches
            scale (int): Downsampling scale factor
            train (bool): If True, create dataset from training folder, else from validation folder
            config (Config): Configuration object
        """
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        self.config = config if config else Config()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Get image paths
        self.subset = "train" if train else "val"
        folder = f"WIDER_{self.subset}"
        self.image_dir = os.path.join(root_dir, folder, "images")
        
        # Load annotations
        anno_file = os.path.join(root_dir, f"wider_face_split", f"wider_face_{self.subset}_bbx_gt.txt")
        self.annotations = parse_wider_face_annotations(anno_file)
        
        # Normalize paths to match annotation entries
        self.image_paths = []
        self.faces_per_image = []
        
        # Find all images with valid annotations
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.image_dir)
                    
                    # Check if this image has annotations
                    for anno_key in self.annotations:
                        if relative_path.endswith(anno_key) or anno_key.endswith(relative_path):
                            if len(self.annotations[anno_key]) > 0:  # Has valid face annotations
                                self.image_paths.append(full_path)
                                self.faces_per_image.append(len(self.annotations[anno_key]))
                                break
        
        print(f"Found {len(self.image_paths)} images with {sum(self.faces_per_image)} faces in {folder} set")

    def __len__(self):
        return len(self.image_paths)
    
    def apply_realistic_degradation(self, img):
        """Apply realistic image degradations to simulate real-world LR images"""
        img_np = np.array(img)
        
        # Apply gaussian blur
        if self.config.add_blur:
            kernel_size = self.config.blur_kernel_size
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
        # Add random noise
        if self.config.add_noise:
            noise = np.random.normal(0, self.config.noise_level, img_np.shape).astype(np.uint8)
            img_np = np.clip(img_np + noise, 0, 255)
        
        # Apply JPEG compression artifacts
        if self.config.add_compression:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
            _, encimg = cv2.imencode('.jpg', img_np, encode_param)
            img_np = cv2.imdecode(encimg, 1)
        
        return Image.fromarray(img_np)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Find matching annotation key
        relative_path = os.path.relpath(img_path, self.image_dir)
        annotation_key = None
        
        for key in self.annotations:
            if relative_path.endswith(key) or key.endswith(relative_path):
                annotation_key = key
                break
        
        # Read the image
        hr_image = Image.open(img_path).convert('RGB')
        width, height = hr_image.size
        
        # Try to focus on tiny faces if possible
        faces = self.annotations.get(annotation_key, [])
        tiny_faces = [face for face in faces if 
                      self.config.min_face_size <= face[2] <= self.config.max_face_size and
                      self.config.min_face_size <= face[3] <= self.config.max_face_size]
        
        # If tiny faces exist, try to crop around one of them
        if tiny_faces and random.random() < 0.7:  # 70% chance to focus on tiny faces
            # Randomly select a tiny face
            face = random.choice(tiny_faces)
            x, y, w, h = face
            
            # Determine crop area with some context around the face
            context_factor = random.uniform(2.0, 4.0)  # Random context
            context_w = w * context_factor
            context_h = h * context_factor
            
            # Center of the face
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Calculate crop boundaries ensuring we don't go out of bounds
            half_size = self.hr_size / 2
            left = max(0, min(width - self.hr_size, center_x - half_size))
            top = max(0, min(height - self.hr_size, center_y - half_size))
            
            # If image is too small, resize it
            if width < self.hr_size or height < self.hr_size:
                ratio = max(self.hr_size / width, self.hr_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                hr_image = hr_image.resize((new_width, new_height), Image.BICUBIC)
                width, height = hr_image.size
                
                # Recalculate crop boundaries
                left = max(0, min(width - self.hr_size, center_x * ratio - half_size))
                top = max(0, min(height - self.hr_size, center_y * ratio - half_size))
            
            # Crop the image
            hr_image = hr_image.crop((left, top, left + self.hr_size, top + self.hr_size))
        else:
            # Random crop if no suitable tiny face or by chance
            if width < self.hr_size or height < self.hr_size:
                # Resize if too small
                ratio = max(self.hr_size / width, self.hr_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                hr_image = hr_image.resize((new_width, new_height), Image.BICUBIC)
                width, height = hr_image.size
            
            # Random crop
            left = random.randint(0, width - self.hr_size)
            top = random.randint(0, height - self.hr_size)
            hr_image = hr_image.crop((left, top, left + self.hr_size, top + self.hr_size))
        
        # Create LR image with realistic degradation
        lr_image_clean = hr_image.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_image = self.apply_realistic_degradation(lr_image_clean)
        
        # Convert to tensors
        hr_tensor = self.transform(hr_image)
        lr_tensor = self.transform(lr_image)
        
        return {
            'lr': lr_tensor, 
            'hr': hr_tensor, 
            'path': img_path
        }

# Edge detection for face detail preservation
class EdgeDetectionLoss(nn.Module):
    def __init__(self, device):
        super(EdgeDetectionLoss, self).__init__()
        self.device = device
        
        # Sobel edge detection kernels
        self.sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        self.sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        
        self.mse_loss = nn.MSELoss()
    
    def detect_edges(self, x):
        # Convert to grayscale by averaging channels
        gray = torch.mean(x, dim=1, keepdim=True)
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Calculate edge magnitude
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        return edge
    
    def forward(self, sr, hr):
        sr_edges = self.detect_edges(sr)
        hr_edges = self.detect_edges(hr)
        return self.mse_loss(sr_edges, hr_edges)

# Perceptual loss with VGG features
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

# SSIM calculation for validation
def calculate_ssim(sr_img, hr_img):
    """Calculate SSIM between super-resolved and high-resolution images"""
    # Convert from tensor to numpy
    if isinstance(sr_img, torch.Tensor):
        sr_img = (sr_img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5
        sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)
    if isinstance(hr_img, torch.Tensor):
        hr_img = (hr_img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5
        hr_img = np.clip(hr_img, 0, 255).astype(np.uint8)
    
    # Convert to grayscale for SSIM
    sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
    hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
    
    return ssim(sr_gray, hr_gray, data_range=255)

# Enhanced trainer with improved metrics and face-focused training
class ESRGANTrainer:
    def __init__(self, config):
        self.config = config
        
        # Create output directories
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.sample_dir, exist_ok=True)
        
        # Initialize networks - use enhanced model for tiny faces
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
        self.edge_loss = EdgeDetectionLoss(config.device)
        
        # Create datasets with config
        self.train_dataset = FaceDataset(
            root_dir=config.wider_face_path,
            hr_size=config.hr_size,
            scale=config.scale_factor,
            train=True,
            config=config
        )
        
        self.val_dataset = FaceDataset(
            root_dir=config.wider_face_path,
            hr_size=config.hr_size,
            scale=config.scale_factor,
            train=False,
            config=config
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
        best_ssim = 0
        
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
                edge_loss = self.edge_loss(sr_imgs, hr_imgs)
                
                # Total loss with weighted components
                # We prioritize edge preservation for facial features
                total_loss = pixel_loss + 0.1 * p_loss + 0.05 * edge_loss
                
                # Backward pass and optimize
                total_loss.backward()
                self.optimizer_G.step()
                
                # Update progress bar
                pbar.set_description(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, " 
                    f"Loss: {total_loss.item():.4f}, "
                    f"Pixel: {pixel_loss.item():.4f}, "
                    f"Percep: {p_loss.item():.4f}, "
                    f"Edge: {edge_loss.item():.4f}"
                )
            
            # Validation
            if (epoch + 1) % self.config.save_interval == 0:
                avg_psnr, avg_ssim = self.validate()
                print(f"Epoch {epoch+1} - Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
                
                # Save model
                self.save_model(epoch + 1)
                
                # Save sample images
                self.save_samples(epoch + 1)
                
                # Save best model (using both PSNR and SSIM)
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    self.save_model(epoch + 1, is_best=True, metric="psnr")
                
                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                    self.save_model(epoch + 1, is_best=True, metric="ssim")
    
    def validate(self):
        self.generator.eval()
        psnr_values = []
        ssim_values = []
        
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
                
                # Calculate SSIM
                ssim_val = calculate_ssim(sr_imgs[0], hr_imgs[0])
                ssim_values.append(ssim_val)
        
        return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)
    
    def save_model(self, epoch, is_best=False, metric=None):
        """Save the model state"""
        if is_best:
            if metric:
                path = os.path.join(self.config.models_dir, f"best_model_{metric}.pth")
            else:
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
            
            # Calculate and display metrics
            psnr_val = compare_psnr(hr_img, sr_img, data_range=255)
            ssim_val = ssim(
                cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY),
                data_range=255
            )
            
            # Save images
            cv2.imwrite(os.path.join(self.config.sample_dir, f"epoch_{epoch}_lr.png"), 
                        cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.config.sample_dir, f"epoch_{epoch}_hr.png"), 
                        cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.config.sample_dir, f"epoch_{epoch}_sr_psnr_{psnr_val:.2f}_ssim_{ssim_val:.4f}.png"), 
                        cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

# Function to upscale all images in a directory
def upscale_directory(model_path, input_dir, output_dir, scale_factor=4, batch_size=4):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet_TinyFaces(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(device)
    
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

# Integrate face detection for evaluation

def detect_faces_and_evaluate(model_path, test_dir, annotation_file=None):
    """Evaluate face detection performance on upscaled images vs original images"""
    # Create output directory for upscaled images
    upscaled_dir = os.path.join(os.path.dirname(test_dir), "upscaled_test")
    os.makedirs(upscaled_dir, exist_ok=True)
    
    # Upscale all test images
    upscale_directory(model_path, test_dir, upscaled_dir)
    
    # Initialize face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Get all image paths
    original_paths = glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
    upscaled_paths = glob.glob(os.path.join(upscaled_dir, "**", "*.jpg"), recursive=True)
    
    # Sort paths to ensure correspondence
    original_paths.sort()
    upscaled_paths.sort()
    
    # Detection statistics
    original_detections = 0
    upscaled_detections = 0
    total_faces = 0
    
    # Parse annotation file if provided
    annotations = {}
    if annotation_file:
        annotations = parse_wider_face_annotations(annotation_file)
    
    # Process each image pair
    for orig_path, up_path in tqdm(zip(original_paths, upscaled_paths), desc="Evaluating face detection"):
        # Read images
        orig_img = cv2.imread(orig_path)
        up_img = cv2.imread(up_path)
        
        # Detect faces in original image
        orig_faces = face_detector.detectMultiScale(
            cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(5, 5)  # Small size to detect tiny faces
        )
        
        # Detect faces in upscaled image
        up_faces = face_detector.detectMultiScale(
            cv2.cvtColor(up_img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)  # Scaled up size (4x)
        )
        
        # Update statistics
        original_detections += len(orig_faces)
        upscaled_detections += len(up_faces)
        
        # If annotation file is provided, calculate ground truth faces
        if annotation_file:
            # Get relative path for lookup in annotations
            rel_path = os.path.relpath(orig_path, test_dir)
            # Find annotations for this image
            for key in annotations:
                if rel_path.endswith(key) or key.endswith(rel_path):
                    # Count tiny faces in ground truth
                    tiny_faces = [face for face in annotations[key] if 
                                 face[2] <= 32 and face[3] <= 32]  # w,h <= 32
                    total_faces += len(tiny_faces)
                    break
        
        # Draw faces on images for visualization (optional)
        for (x, y, w, h) in orig_faces:
            cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        for (x, y, w, h) in up_faces:
            cv2.rectangle(up_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save visualization
        vis_dir = os.path.join(os.path.dirname(test_dir), "face_detection_vis")
        os.makedirs(vis_dir, exist_ok=True)
        base_name = os.path.basename(orig_path)
        cv2.imwrite(os.path.join(vis_dir, f"original_{base_name}"), orig_img)
        cv2.imwrite(os.path.join(vis_dir, f"upscaled_{base_name}"), up_img)
    
    # Calculate detection rate
    print(f"Original detection count: {original_detections}")
    print(f"Upscaled detection count: {upscaled_detections}")
    
    if annotation_file:
        print(f"Ground truth tiny faces: {total_faces}")
        print(f"Original detection rate: {original_detections/total_faces:.4f}")
        print(f"Upscaled detection rate: {upscaled_detections/total_faces:.4f}")
    
    improvement = (upscaled_detections - original_detections) / max(1, original_detections) * 100
    print(f"Detection improvement: {improvement:.2f}%")
    
    return {
        "original_detections": original_detections,
        "upscaled_detections": upscaled_detections,
        "total_faces": total_faces,
        "improvement_percentage": improvement
    }
# Add new function to use a more advanced face detector (MTCNN)
def evaluate_with_mtcnn(model_path, test_dir, annotation_file=None):
    """Evaluate using MTCNN face detector which works better for tiny faces"""
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError:
        print("Please install facenet-pytorch: pip install facenet-pytorch")
        return
    
    # Create output directory for upscaled images
    upscaled_dir = os.path.join(os.path.dirname(test_dir), "upscaled_test")
    os.makedirs(upscaled_dir, exist_ok=True)
    
    # Upscale all test images if not already done
    if not os.path.exists(upscaled_dir) or len(os.listdir(upscaled_dir)) == 0:
        upscale_directory(model_path, test_dir, upscaled_dir)
    
    # Initialize MTCNN face detector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For original images (lower confidence for tiny faces)
    mtcnn_original = MTCNN(
        image_size=160, 
        margin=0, 
        min_face_size=10,
        thresholds=[0.6, 0.7, 0.7],  # Lower thresholds for tiny faces
        factor=0.709, 
        device=device
    )
    
    # For upscaled images
    mtcnn_upscaled = MTCNN(
        image_size=160, 
        margin=0, 
        min_face_size=40,  # 4x larger for upscaled images
        thresholds=[0.7, 0.8, 0.8],  # Standard thresholds
        factor=0.709, 
        device=device
    )
    
    # Get all image paths
    original_paths = glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
    upscaled_paths = glob.glob(os.path.join(upscaled_dir, "**", "*.jpg"), recursive=True)
    
    # Sort paths to ensure correspondence
    original_paths.sort()
    upscaled_paths.sort()
    
    # Detection statistics
    original_detections = 0
    upscaled_detections = 0
    total_tiny_faces = 0
    
    # Process each image pair
    results = []
    
    for orig_path, up_path in tqdm(zip(original_paths, upscaled_paths), desc="Evaluating face detection with MTCNN"):
        # Read images
        orig_img = cv2.imread(orig_path)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        up_img = cv2.imread(up_path)
        up_img_rgb = cv2.cvtColor(up_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        orig_pil = Image.fromarray(orig_img_rgb)
        up_pil = Image.fromarray(up_img_rgb)
        
        # Detect faces
        orig_boxes, orig_probs = mtcnn_original.detect(orig_pil)
        up_boxes, up_probs = mtcnn_upscaled.detect(up_pil)
        
        # Count valid detections
        orig_count = 0 if orig_boxes is None else len(orig_boxes)
        up_count = 0 if up_boxes is None else len(up_boxes)
        
        original_detections += orig_count
        upscaled_detections += up_count
        
        # Parse ground truth if available
        gt_faces = []
        if annotation_file:
            annotations = parse_wider_face_annotations(annotation_file)
            rel_path = os.path.relpath(orig_path, test_dir)
            
            for key in annotations:
                if rel_path.endswith(key) or key.endswith(rel_path):
                    gt_faces = annotations[key]
                    # Count tiny faces in ground truth
                    tiny_faces = [face for face in gt_faces if 
                                 face[2] <= 32 and face[3] <= 32]  # w,h <= 32
                    total_tiny_faces += len(tiny_faces)
                    break
        
        # Draw faces on images for visualization
        orig_vis = orig_img.copy()
        up_vis = up_img.copy()
        
        # Draw original detections
        if orig_boxes is not None:
            for box in orig_boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(orig_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw upscaled detections
        if up_boxes is not None:
            for box in up_boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(up_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ground truth boxes on original if available
        if gt_faces:
            for box in gt_faces:
                x, y, w, h = [int(b) for b in box]
                cv2.rectangle(orig_vis, (x, y), (x+w, y+h), (255, 0, 0), 1)  # Red for ground truth
        
        # Save visualization
        vis_dir = os.path.join(os.path.dirname(test_dir), "mtcnn_detection_vis")
        os.makedirs(vis_dir, exist_ok=True)
        base_name = os.path.basename(orig_path)
        cv2.imwrite(os.path.join(vis_dir, f"original_{base_name}"), orig_vis)
        cv2.imwrite(os.path.join(vis_dir, f"upscaled_{base_name}"), up_vis)
        
        # Store result for this image
        results.append({
            'image': os.path.basename(orig_path),
            'original_detections': orig_count,
            'upscaled_detections': up_count,
            'ground_truth_faces': len(gt_faces) if gt_faces else 'N/A'
        })
    
    # Calculate detection rate
    print(f"Original detection count: {original_detections}")
    print(f"Upscaled detection count: {upscaled_detections}")
    
    if annotation_file and total_tiny_faces > 0:
        print(f"Ground truth tiny faces: {total_tiny_faces}")
        print(f"Original detection rate: {original_detections/total_tiny_faces:.4f}")
        print(f"Upscaled detection rate: {upscaled_detections/total_tiny_faces:.4f}")
    
    improvement = (upscaled_detections - original_detections) / max(1, original_detections) * 100
    print(f"Detection improvement: {improvement:.2f}%")
    
    # Save detailed results
    with open(os.path.join(os.path.dirname(test_dir), "mtcnn_detection_results.json"), 'w') as f:
        json.dump({
            'summary': {
                'original_detections': original_detections,
                'upscaled_detections': upscaled_detections,
                'total_tiny_faces': total_tiny_faces,
                'improvement_percentage': improvement
            },
            'per_image': results
        }, f, indent=4)
    
    return {
        'original_detections': original_detections,
        'upscaled_detections': upscaled_detections,
        'total_tiny_faces': total_tiny_faces,
        'improvement_percentage': improvement
    }

# Add a face-focused training method with additional enhancements
def train_with_face_focus(config):
    """Train ESRGAN with additional focus on facial features and tiny faces"""
    # Initialize trainer
    trainer = ESRGANTrainer(config)
    
    # Add face detector for additional loss
    try:
        from facenet_pytorch import MTCNN
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        face_detector = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=10,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709, 
            device=device,
            keep_all=True
        )
        use_face_detector = True
        print("Using MTCNN for face-aware training")
    except ImportError:
        use_face_detector = False
        print("MTCNN not available, using default training method")
    
    # Create an adaptive learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer_G, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    best_psnr = 0
    best_ssim = 0
    
    for epoch in range(config.num_epochs):
        trainer.generator.train()
        epoch_losses = []
        
        pbar = tqdm(trainer.train_loader)
        for batch in pbar:
            # Get data
            lr_imgs = batch['lr'].to(config.device)
            hr_imgs = batch['hr'].to(config.device)
            
            # Forward pass
            trainer.optimizer_G.zero_grad()
            sr_imgs = trainer.generator(lr_imgs)
            
            # Calculate base losses
            pixel_loss = trainer.l1_loss(sr_imgs, hr_imgs)
            p_loss = trainer.perceptual_loss(sr_imgs, hr_imgs)
            edge_loss = trainer.edge_loss(sr_imgs, hr_imgs)
            
            # Calculate face-focused loss if available
            face_loss = 0
            if use_face_detector:
                # Process batch to find faces
                face_regions_hr = []
                face_regions_sr = []
                
                for i in range(lr_imgs.size(0)):
                    # Convert to numpy for face detection
                    hr_img = ((hr_imgs[i].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    hr_pil = Image.fromarray(hr_img)
                    
                    # Detect faces in HR image
                    boxes, _ = face_detector.detect(hr_pil)
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = [int(b) for b in box]
                            
                            # Extract face region from both HR and SR
                            if x1 < x2 and y1 < y2:
                                # Ensure coordinates are within image bounds
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(hr_imgs.size(3), x2)
                                y2 = min(hr_imgs.size(2), y2)
                                
                                if x2 > x1 and y2 > y1:
                                    face_hr = hr_imgs[i, :, y1:y2, x1:x2]
                                    face_sr = sr_imgs[i, :, y1:y2, x1:x2]
                                    
                                    if face_hr.size(1) > 0 and face_hr.size(2) > 0:
                                        face_regions_hr.append(face_hr)
                                        face_regions_sr.append(face_sr)
                
                # Calculate face-specific loss if faces were found
                if face_regions_hr:
                    # Stack regions into batches
                    face_batch_hr = torch.stack(face_regions_hr)
                    face_batch_sr = torch.stack(face_regions_sr)
                    
                    # Calculate losses specifically for face regions
                    face_pixel_loss = trainer.l1_loss(face_batch_sr, face_batch_hr)
                    face_edge_loss = trainer.edge_loss(face_batch_sr, face_batch_hr)
                    
                    # Add to face loss with higher weight
                    face_loss = 2.0 * face_pixel_loss + 1.0 * face_edge_loss
            
            # Total loss with weighted components
            # Priority: face details > edges > pixels > perceptual
            total_loss = pixel_loss + 0.1 * p_loss + 0.5 * edge_loss
            
            # Add face loss if available
            if face_loss != 0:
                total_loss += 1.0 * face_loss
            
            # Backward pass and optimize
            total_loss.backward()
            trainer.optimizer_G.step()
            
            # Store loss for scheduler
            epoch_losses.append(total_loss.item())
            
            # Update progress bar
            face_loss_val = face_loss if isinstance(face_loss, float) else face_loss.item()
            pbar.set_description(
                f"Epoch {epoch+1}/{config.num_epochs}, " 
                f"Loss: {total_loss.item():.4f}, "
                f"Pixel: {pixel_loss.item():.4f}, "
                f"Face: {face_loss_val:.4f}"
            )
        
        # Update learning rate based on average loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        scheduler.step(avg_loss)
        
        # Validation
        if (epoch + 1) % config.save_interval == 0:
            avg_psnr, avg_ssim = trainer.validate()
            print(f"Epoch {epoch+1} - Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
            
            # Save model
            trainer.save_model(epoch + 1)
            
            # Save sample images
            trainer.save_samples(epoch + 1)
            
            # Save best model (using both PSNR and SSIM)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                trainer.save_model(epoch + 1, is_best=True, metric="psnr")
            
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                trainer.save_model(epoch + 1, is_best=True, metric="ssim")

# Add an inference helper for the model
def apply_model_to_image(model_path, input_image_path, output_image_path=None):
    """Apply the trained ESRGAN model to a single image"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet_TinyFaces(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load and process image
    img = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate SR image
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to numpy and adjust range
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Convert to BGR for OpenCV
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    # Save or return
    if output_image_path:
        cv2.imwrite(output_image_path, output_bgr)
        print(f"Saved super-resolved image to {output_image_path}")
        return output_image_path
    else:
        return output

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="ESRGAN for Tiny Face Enhancement")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "upscale"], 
                        help="Operation mode")
    parser.add_argument("--data_path", type=str, help="Path to WiderFace dataset")
    parser.add_argument("--model_path", type=str, help="Path to saved model (for upscale/evaluate)")
    parser.add_argument("--input_dir", type=str, help="Input directory (for upscale)")
    parser.add_argument("--output_dir", type=str, help="Output directory (for upscale)")
    parser.add_argument("--anno_file", type=str, help="Path to annotation file (for evaluate)")
    parser.add_argument("--test_dir", type=str, help="Test directory (for evaluate)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--face_focus", action="store_true", help="Use face-focused training")
    parser.add_argument("--use_mtcnn", action="store_true", help="Use MTCNN for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.data_path:
            parser.error("--data_path required for training mode")
        
        # Setup configuration
        config = Config()
        config.wider_face_path = args.data_path
        config.batch_size = args.batch_size
        config.num_epochs = args.epochs
        config.lr = args.lr
        
        # Start training
        if args.face_focus:
            train_with_face_focus(config)
        else:
            trainer = ESRGANTrainer(config)
            trainer.train()
    
    elif args.mode == "evaluate":
        if not args.model_path or not args.test_dir:
            parser.error("--model_path and --test_dir required for evaluate mode")
        
        # Start evaluation
        if args.use_mtcnn:
            evaluate_with_mtcnn(args.model_path, args.test_dir, args.anno_file)
        else:
            detect_faces_and_evaluate(args.model_path, args.test_dir, args.anno_file)
    
    elif args.mode == "upscale":
        if not args.model_path or not args.input_dir or not args.output_dir:
            parser.error("--model_path, --input_dir, and --output_dir required for upscale mode")
        
        # Start upscaling
        upscale_directory(args.model_path, args.input_dir, args.output_dir)

if __name__ == "__main__":
    import torch.nn.functional as F  # Required import for RRDBNet forward function
    main()