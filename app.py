import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import zipfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import io

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=32):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.labels_path = os.path.join(root_dir, "labels.json")
        
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)
        
        # Filter out labels that are too long
        self.labels = {k: v for k, v in self.labels.items() if len(v) <= max_length}
        self.image_files = list(self.labels.keys())
        
        # Create character vocabulary
        all_chars = set()
        for label in self.labels.values():
            all_chars.update(label)
        
        # Add special tokens
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(sorted(all_chars)):
            self.char_to_idx[char] = i + 2
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = self.labels[img_name]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to indices
        label_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in label]
        
        return {
            'image': image,
            'text': label_indices,
            'length': len(label_indices),
            'raw_text': label
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    max_len = max(lengths)
    padded_texts = []
    
    for text in texts:
        padded = text + [0] * (max_len - len(text))  # 0 is <PAD>
        padded_texts.append(padded)
    
    padded_texts = torch.tensor(padded_texts, dtype=torch.long)
    
    return {
        'image': images,
        'text': padded_texts,
        'length': torch.tensor(lengths, dtype=torch.long)
    }

class ImprovedTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super(ImprovedTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=0.3,
            bidirectional=True
        )
        self.projection = nn.Linear(hidden_size * 2, hidden_size)  # Bidirectional
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        output, (hidden, cell) = self.lstm(packed)
        
        # Use final hidden state from both directions
        hidden = hidden.view(hidden.size(1), -1)  # Concatenate bidirectional
        hidden = self.projection(hidden)
        
        return hidden

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim, text_embed_size, img_channels=1):
        super(ImprovedGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Project and reshape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + text_embed_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(True)
        )
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final layer
            nn.ConvTranspose2d(64, img_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, noise, text_embed):
        x = torch.cat([noise, text_embed], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.conv_blocks(x)
        return x

class ImprovedDiscriminator(nn.Module):
    def __init__(self, text_embed_size, img_channels=1):
        super(ImprovedDiscriminator, self).__init__()
        
        # Image processing path
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            SpectralNorm(nn.Conv2d(img_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            SpectralNorm(nn.Linear(512 * 4 * 4 + text_embed_size, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            SpectralNorm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img, text_embed):
        conv_out = self.conv_blocks(img)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Concatenate image and text features
        combined = torch.cat([conv_out, text_embed], dim=1)
        validity = self.classifier(combined)
        
        return validity

class ImprovedHandwritingGAN(nn.Module):
    def __init__(self, latent_dim, vocab_size, embed_size, hidden_size):
        super(ImprovedHandwritingGAN, self).__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        self.encoder = ImprovedTextEncoder(vocab_size, embed_size, hidden_size)
        self.generator = ImprovedGenerator(latent_dim, hidden_size)
        self.discriminator = ImprovedDiscriminator(hidden_size)
        
        # Optimizers with different learning rates
        self.g_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.encoder.parameters()),
            lr=0.0001, betas=(0.0, 0.9)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0004, betas=(0.0, 0.9)
        )
        
        # Loss functions
        self.adversarial_loss = nn.MSELoss()  # LSGAN loss
        
        # For gradient penalty
        self.lambda_gp = 10
    
    def compute_gradient_penalty(self, real_samples, fake_samples, text_embed):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random weight term for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates, text_embed)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_images, captions, lengths):
        batch_size = real_images.size(0)
        device = real_images.device
        
        # Labels for adversarial loss
        valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
        fake = torch.zeros(batch_size, 1, device=device, requires_grad=False)
        
        # Encode text
        text_features = self.encoder(captions, lengths)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_validity = self.discriminator(real_images, text_features.detach())
        d_real_loss = self.adversarial_loss(real_validity, valid)
        
        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(noise, text_features.detach())
        fake_validity = self.discriminator(fake_images.detach(), text_features.detach())
        d_fake_loss = self.adversarial_loss(fake_validity, fake)
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(real_images, fake_images, text_features.detach())
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator (every other iteration)
        if hasattr(self, 'train_step_count'):
            self.train_step_count += 1
        else:
            self.train_step_count = 0
            
        if self.train_step_count % 2 == 0:
            self.g_optimizer.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.latent_dim, device=device)
            fake_images = self.generator(noise, text_features)
            fake_validity = self.discriminator(fake_images, text_features)
            
            # Generator loss
            g_loss = self.adversarial_loss(fake_validity, valid)
            
            g_loss.backward()
            self.g_optimizer.step()
        else:
            g_loss = torch.tensor(0.0)
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_real': d_real_loss.item(),
            'd_fake': d_fake_loss.item(),
            'gp': gp.item()
        }
    
    def generate_samples(self, text, char_to_idx, num_samples=1):
        self.eval()
        with torch.no_grad():
            # Convert text to indices
            text_indices = [char_to_idx.get(c, char_to_idx.get('<UNK>', 1)) for c in text]
            text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0)
            lengths = torch.tensor([len(text_indices)], dtype=torch.long)
            
            if next(self.parameters()).is_cuda:
                text_tensor = text_tensor.cuda()
                lengths = lengths.cuda()
            
            # Encode text
            text_features = self.encoder(text_tensor, lengths)
            
            # Generate multiple samples
            generated_images = []
            for _ in range(num_samples):
                noise = torch.randn(1, self.latent_dim, device=text_features.device)
                generated_image = self.generator(noise, text_features)
                
                # Convert to numpy
                generated_image = (generated_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
                generated_image = torch.clamp(generated_image, 0, 1)
                generated_image = generated_image.cpu().numpy()[0, 0]
                generated_images.append(generated_image)
            
        self.train()
        return generated_images[0] if num_samples == 1 else generated_images

class StreamlitHandwritingApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'dataset' not in st.session_state:
            st.session_state.dataset = None
        if 'char_to_idx' not in st.session_state:
            st.session_state.char_to_idx = None
        if 'hyperparams' not in st.session_state:
            st.session_state.hyperparams = {
                'latent_dim': 100,
                'embed_size': 256,
                'hidden_size': 512,
                'batch_size': 16,
                'learning_rate': 0.0001
            }
            
        self.hyperparams = st.session_state.hyperparams
    
    def setup_folders(self):
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('generated', exist_ok=True)
    
    def process_uploaded_dataset(self, uploaded_file):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, 'dataset.zip')
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('uploads')
                
                data_dir = Path('uploads')
                if not (data_dir / 'labels.json').exists():
                    st.error("Dataset must contain a labels.json file!")
                    return False
                
                return True
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            return False
    
    def load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        dataset = HandwritingDataset('uploads', transform=transform, max_length=32)
        st.session_state.dataset = dataset
        st.session_state.char_to_idx = dataset.char_to_idx
        
        return DataLoader(
            dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    def initialize_model(self, vocab_size):
        model = ImprovedHandwritingGAN(
            self.hyperparams['latent_dim'],
            vocab_size,
            self.hyperparams['embed_size'],
            self.hyperparams['hidden_size']
        ).to(self.device)
        return model
    
    def train_model(self, num_epochs, progress_bar, status_text):
        dataloader = self.load_dataset()
        vocab_size = st.session_state.dataset.vocab_size
        
        model = self.initialize_model(vocab_size)
        all_losses = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': [], 'gp': []}
        
        try:
            for epoch in range(num_epochs):
                epoch_losses = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': [], 'gp': []}
                
                for batch_idx, batch in enumerate(dataloader):
                    images = batch['image'].to(self.device)
                    captions = batch['text'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    
                    batch_losses = model.train_step(images, captions, lengths)
                    
                    for key in epoch_losses:
                        epoch_losses[key].append(batch_losses[key])
                    
                    # Update progress within epoch
                    batch_progress = (batch_idx + 1) / len(dataloader)
                    total_progress = (epoch + batch_progress) / num_epochs
                    progress_bar.progress(total_progress)
                
                # Calculate epoch averages
                epoch_avg = {key: np.mean(values) for key, values in epoch_losses.items()}
                for key in all_losses:
                    all_losses[key].append(epoch_avg[key])
                
                # Update status
                status_text.text(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"D_Loss: {epoch_avg['d_loss']:.4f}, "
                    f"G_Loss: {epoch_avg['g_loss']:.4f}, "
                    f"GP: {epoch_avg['gp']:.4f}"
                )
                
                # Save model periodically
                if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                    self.save_model(model, f'epoch_{epoch+1}', vocab_size)
            
            return all_losses
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return None
    
    def save_model(self, model, name, vocab_size):
        try:
            checkpoint = {
                'encoder_state_dict': model.encoder.state_dict(),
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'hyperparams': self.hyperparams,
                'vocab_size': vocab_size,
                'char_to_idx': st.session_state.char_to_idx
            }
            torch.save(checkpoint, f'models/checkpoint_{name}.pth')
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    def load_model(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.hyperparams = checkpoint['hyperparams']
            vocab_size = checkpoint['vocab_size']
            
            st.session_state.hyperparams = self.hyperparams
            st.session_state.char_to_idx = checkpoint['char_to_idx']
            
            # Initialize model
            model = ImprovedHandwritingGAN(
                self.hyperparams['latent_dim'],
                vocab_size,
                self.hyperparams['embed_size'],
                self.hyperparams['hidden_size']
            ).to(self.device)
            
            # Load state dicts
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.generator.load_state_dict(checkpoint['generator_state_dict'])
            model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # Store in session state
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_handwriting(self, text, num_samples=1):
        if st.session_state.model is None or st.session_state.char_to_idx is None:
            return None
            
        model = st.session_state.model
        char_to_idx = st.session_state.char_to_idx
        
        return model.generate_samples(text, char_to_idx, num_samples)

def main():
    st.set_page_config(
        page_title="Improved Handwriting GAN",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Improved Deep Learning Handwriting Generation")
    st.write("""
    This improved version uses advanced GAN techniques including:
    - **Spectral Normalization** for training stability
    - **Gradient Penalty** for better convergence
    - **Bidirectional LSTM** for better text understanding
    - **Improved Architecture** for higher quality generation
    """)
    
    # Display device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        st.sidebar.success(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.info("💻 Using CPU (GPU recommended for training)")
    
    app = StreamlitHandwritingApp()
    app.setup_folders()
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Dataset", "Train Model", "Generate Handwriting"],
        index=0
    )
    
    if page == "Upload Dataset":
        st.header("📁 Upload Dataset")
        st.write("""
        Upload a ZIP file containing:
        1. **Images** of handwritten text (PNG, JPG, etc.) - preferably 64x64 or higher resolution
        2. **labels.json** file mapping image filenames to text content
        """)
        
        # Show example format
        with st.expander("📋 Dataset Format Example"):
            st.code('''
Dataset Structure:
dataset.zip
├── image1.png
├── image2.png
├── image3.png
└── labels.json

labels.json format:
{
  "image1.png": "Hello World",
  "image2.png": "Machine Learning",
  "image3.png": "Deep Learning"
}

Tips for better results:
- Use high-quality, clear handwriting images
- Consistent image sizes work better
- Shorter text labels (5-15 characters) train faster
- More diverse handwriting styles = better generalization
            ''', language='json')
        
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
        if uploaded_file is not None:
            if app.process_uploaded_dataset(uploaded_file):
                st.success("✅ Dataset uploaded and processed successfully!")
                try:
                    dataloader = app.load_dataset()
                    dataset = st.session_state.dataset
                    
                    st.write(f"📊 Found **{len(dataset)}** samples")
                    st.write(f"🔤 Vocabulary size: **{dataset.vocab_size}** characters")
                    
                    # Show vocabulary
                    vocab_chars = list(dataset.char_to_idx.keys())
                    st.write(f"📝 Characters: {', '.join(vocab_chars[:20])}{'...' if len(vocab_chars) > 20 else ''}")
                    
                    # Show sample data
                    sample_batch = next(iter(dataloader))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Batch Size", sample_batch['image'].shape[0])
                        st.metric("Image Shape", f"{sample_batch['image'].shape[2]}×{sample_batch['image'].shape[3]}")
                    with col2:
                        st.metric("Total Batches", len(dataloader))
                        st.metric("Max Text Length", max(sample_batch['length']).item())
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
    
    elif page == "Train Model":
        st.header("🏋️ Train Model")
        
        if not os.path.exists('uploads/labels.json'):
            st.error("❌ Please upload a dataset first!")
            st.info("👈 Go to 'Upload Dataset' page to upload your training data")
            return
        
        # Load dataset info
        try:
            dataloader = app.load_dataset()
            dataset = st.session_state.dataset
            st.success(f"✅ Dataset ready: {len(dataset)} samples, {dataset.vocab_size} unique characters")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return
        
        # Training parameters
        st.subheader("⚙️ Training Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.slider("🔄 Number of epochs", 10, 200, 50,
                                 help="More epochs = better quality but longer training")
            batch_size = st.selectbox("📦 Batch size", [8, 16, 32], index=1,
                                    help="Smaller batch = more stable training")
            learning_rate = st.selectbox("📈 Learning rate", [0.0001, 0.0002], index=0)
        
        with col2:
            latent_dim = st.selectbox("🎲 Latent dimension", [64, 100, 128], index=1,
                                    help="Higher = more complex generations")
            hidden_size = st.selectbox("🧠 Hidden size", [256, 512, 768], index=1,
                                     help="Larger = more model capacity")
            embed_size = st.selectbox("📝 Embedding size", [128, 256, 384], index=1)
        
        # Update hyperparameters
        app.hyperparams.update({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'latent_dim': latent_dim,
            'hidden_size': hidden_size,
            'embed_size': embed_size
        })
        st.session_state.hyperparams = app.hyperparams
        
        # Training info
        estimated_time = (num_epochs * len(dataset) // batch_size) * (3 if device.type == 'cpu' else 1)
        st.info(f"⏱️ Estimated training time: ~{estimated_time:.1f} minutes")
        
        # Training tips
        with st.expander("💡 Training Tips"):
            st.write("""
            **For Better Results:**
            - Start with 50-100 epochs for initial testing
            - Use smaller batch sizes (8-16) for more stable training
            - Monitor the losses - they should decrease over time
            - Discriminator loss should stay between 0.3-0.8
            - Generator loss should gradually decrease
            - Save models every 10 epochs for checkpoints
            
            **If Training Fails:**
            - Reduce batch size to 8
            - Lower learning rate to 0.0001
            - Check dataset quality and labels
            """)
        
        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔥 Training in progress..."):
                losses = app.train_model(num_epochs, progress_bar, status_text)
            
            if losses:
                st.success("🎉 Training completed successfully!")
                
                # Plot training progress
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                epochs = range(1, len(losses['d_loss']) + 1)
                
                # Discriminator vs Generator Loss
                ax1.plot(epochs, losses['d_loss'], 'r-', label='Discriminator Loss', linewidth=2)
                ax1.plot(epochs, losses['g_loss'], 'b-', label='Generator Loss', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Discriminator vs Generator Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Discriminator Components
                ax2.plot(epochs, losses['d_real'], 'g-', label='D Real Loss', linewidth=2)
                ax2.plot(epochs, losses['d_fake'], 'orange', label='D Fake Loss', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Discriminator Components')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Gradient Penalty
                ax3.plot(epochs, losses['gp'], 'purple', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Gradient Penalty')
                ax3.set_title('Gradient Penalty Over Time')
                ax3.grid(True, alpha=0.3)
                
                # Combined view
                ax4.plot(epochs, losses['d_loss'], 'r-', alpha=0.7, label='D Loss')
                ax4.plot(epochs, losses['g_loss'], 'b-', alpha=0.7, label='G Loss')
                ax4.fill_between(epochs, losses['d_loss'], alpha=0.3, color='red')
                ax4.fill_between(epochs, losses['g_loss'], alpha=0.3, color='blue')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss')
                ax4.set_title('Training Progress Overview')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.balloons()
                st.info("💾 Models saved automatically every 10 epochs in the 'models' folder")
    
    elif page == "Generate Handwriting":
        st.header("✨ Generate Handwritten Text")
        
        model_files = []
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        
        if not model_files:
            st.error("❌ No trained models found!")
            st.info("👈 Go to 'Train Model' page to train your first model")
            return
        
        # Model selection and status
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model = st.selectbox("🤖 Select a model", 
                                        sorted(model_files, reverse=True),
                                        help="Choose a trained model checkpoint")
        
        with col2:
            if st.session_state.model_loaded:
                st.success("✅ Model Ready")
            else:
                st.warning("⚠️ No Model Loaded")
        
        # Load model button
        if st.button("📥 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                if app.load_model(f'models/{selected_model}'):
                    st.success("✅ Model loaded successfully!")
                    
                    # Show model info
                    if st.session_state.char_to_idx:
                        vocab_chars = list(st.session_state.char_to_idx.keys())
                        st.write(f"🔤 Model vocabulary: {len(vocab_chars)} characters")
                        st.write(f"📝 Supported characters: {', '.join(vocab_chars[:30])}{'...' if len(vocab_chars) > 30 else ''}")
                    
                    st.rerun()
        
        if st.session_state.model_loaded and st.session_state.char_to_idx:
            # Text input and generation
            st.subheader("📝 Enter Text to Generate")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                text = st.text_input("Text input", "Hello World!", 
                                   help="Enter the text you want to convert to handwriting")
            with col2:
                st.metric("Text Length", len(text))
            
            # Check character support
            if text:
                char_to_idx = st.session_state.char_to_idx
                unsupported_chars = [c for c in text if c not in char_to_idx and c != ' ']
                
                if unsupported_chars:
                    st.warning(f"⚠️ Unsupported characters: {', '.join(set(unsupported_chars))}")
                    st.write("These will be replaced with <UNK> token")
                else:
                    st.success("✅ All characters are supported!")
            
            # Generation options
            col1, col2 = st.columns(2)
            with col1:
                num_samples = st.slider("🎨 Number of samples", 1, 5, 1,
                                      help="Generate multiple variations")
            with col2:
                show_process = st.checkbox("🔍 Show generation process", False)
            
            # Generation tips
            with st.expander("💡 Generation Tips"):
                st.write("""
                **For Better Results:**
                - Use text similar to your training data
                - Shorter text (5-15 characters) usually works better
                - Try common words and phrases first
                - Generate multiple samples to see variations
                
                **Supported Characters:**
                - Letters: a-z, A-Z
                - Numbers: 0-9
                - Common punctuation: . , ! ? ' "
                - Special characters depend on your training data
                """)
            
            if st.button("🎨 Generate Handwriting", type="primary"):
                if not text.strip():
                    st.error("❌ Please enter some text to generate")
                    return
                    
                with st.spinner("🎨 Generating handwriting..."):
                    try:
                        if show_process:
                            st.write("🔄 Encoding text...")
                            
                        generated_images = app.generate_handwriting(text, num_samples)
                        
                        if generated_images is not None:
                            if show_process:
                                st.write("✅ Generation complete!")
                            
                            # Display generated images
                            if num_samples == 1:
                                generated_images = [generated_images]
                            
                            for i, generated_image in enumerate(generated_images):
                                st.subheader(f"Generated Sample {i+1}")
                                
                                # Create a high-quality plot
                                fig, ax = plt.subplots(figsize=(12, 4))
                                ax.imshow(generated_image, cmap='gray', interpolation='bilinear')
                                ax.set_title(f'Generated Handwriting: "{text}"', fontsize=16, pad=20)
                                ax.axis('off')
                                ax.set_facecolor('white')
                                fig.patch.set_facecolor('white')
                                
                                # Add border
                                for spine in ax.spines.values():
                                    spine.set_visible(True)
                                    spine.set_linewidth(2)
                                    spine.set_color('lightgray')
                                
                                st.pyplot(fig)
                                
                                # Download button for each sample
                                img_array = (generated_image * 255).astype(np.uint8)
                                img_pil = Image.fromarray(img_array)
                                
                                # Enhance image quality
                                img_pil = img_pil.resize((img_pil.width * 2, img_pil.height * 2), Image.LANCZOS)
                                
                                buf = io.BytesIO()
                                img_pil.save(buf, format='PNG')
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    f"💾 Download Sample {i+1}",
                                    data=byte_im,
                                    file_name=f"handwriting_{text.replace(' ', '_')}_sample_{i+1}.png",
                                    mime="image/png",
                                    key=f"download_{i}"
                                )
                                
                                if i < len(generated_images) - 1:
                                    st.write("---")
                            
                        else:
                            st.error("❌ Failed to generate image")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating handwriting: {str(e)}")
                        st.write("**Debugging info:**")
                        st.write(f"- Text: '{text}'")
                        st.write(f"- Text length: {len(text)}")
                        st.write(f"- Model loaded: {st.session_state.model_loaded}")
                        st.write(f"- Vocabulary available: {st.session_state.char_to_idx is not None}")
    
    # Sidebar info
    with st.sidebar:
        st.write("---")
        st.subheader("📊 System Status")
        st.write(f"**Device**: {device.type.upper()}")
        if device.type == 'cuda':
            st.write(f"**GPU Memory**: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        if st.session_state.model_loaded:
            st.write("**Model Status**: ✅ Loaded")
            if st.session_state.char_to_idx:
                st.write(f"**Vocabulary Size**: {len(st.session_state.char_to_idx)}")
        else:
            st.write("**Model Status**: ❌ Not Loaded")
        
        st.write("---")
        st.subheader("🔧 Troubleshooting")
        st.write("""
        **Common Issues:**
        - **Poor quality**: Train longer (100+ epochs)
        - **Blurry images**: Check dataset quality
        - **Training crashes**: Reduce batch size
        - **Out of memory**: Use smaller model/batch
        """)
        
        st.write("---")
        st.subheader("🔗 Resources")
        st.write("• [GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)")
        st.write("• [Spectral Normalization Paper](https://arxiv.org/abs/1802.05957)")
        st.write("• [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)")

if __name__ == "__main__":
    main()
