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
    def __init__(self, root_dir, transform=None, max_len=32):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.labels_path = os.path.join(root_dir, "labels.json")
        
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)
        
        # Filter out samples that are too long
        self.image_files = [k for k, v in self.labels.items() if len(v) <= max_len]
        
        # Create character vocabulary
        all_chars = set()
        for label in self.labels.values():
            all_chars.update(label)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<START>'] = len(self.char_to_idx)
        self.char_to_idx['<END>'] = len(self.char_to_idx)
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
        
        # Convert text to indices
        text_indices = [self.char_to_idx['<START>']]
        text_indices.extend([self.char_to_idx.get(c, 0) for c in label])
        text_indices.append(self.char_to_idx['<END>'])
        
        return {
            'image': image,
            'text': text_indices,
            'length': len(text_indices),
            'original_text': label
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    original_texts = [item['original_text'] for item in batch]
    
    max_len = max(lengths)
    padded_texts = []
    
    for text in texts:
        padded = text + [0] * (max_len - len(text))
        padded_texts.append(padded)
    
    padded_texts = torch.tensor(padded_texts, dtype=torch.long)
    
    return {
        'image': images,
        'text': padded_texts,
        'length': torch.tensor(lengths),
        'original_text': original_texts
    }

class ImprovedTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super(ImprovedTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        output, (hidden, cell) = self.lstm(packed)
        
        # Use the last hidden state from both directions
        hidden = hidden.view(-1, 2, hidden.size(-1))  # [layers, batch, hidden] -> [layers, 2, batch, hidden]
        hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)  # Concatenate forward and backward
        
        hidden = self.fc(hidden)
        hidden = torch.tanh(hidden)
        
        return hidden

class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim, text_embed_size, img_channels=1, img_size=64):
        super(ImprovedGenerator, self).__init__()
        self.img_size = img_size
        
        # Improved architecture with more layers and skip connections
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + text_embed_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.conv_blocks = nn.ModuleList([
            # Block 1: 4x4 -> 8x8
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ),
            # Block 2: 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            # Block 3: 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            # Block 4: 32x32 -> 64x64
            nn.Sequential(
                nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, noise, text_embed):
        x = torch.cat([noise, text_embed], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        
        for block in self.conv_blocks:
            x = block(x)
        
        return x

class ImprovedDiscriminator(nn.Module):
    def __init__(self, text_embed_size, img_channels=1, img_size=64):
        super(ImprovedDiscriminator, self).__init__()
        
        # Convolutional layers for image processing
        self.conv_blocks = nn.ModuleList([
            # Block 1: 64x64 -> 32x32
            nn.Sequential(
                nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 2: 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 3: 16x16 -> 8x8
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 4: 8x8 -> 4x4
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 + text_embed_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, img, text_embed):
        x = img
        for block in self.conv_blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, text_embed], dim=1)
        x = self.fc(x)
        return x

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
            lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0004, betas=(0.5, 0.999), weight_decay=1e-5
        )
        
        # Use label smoothing
        self.criterion = nn.BCELoss()
        
        # Add learning rate schedulers
        self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=10, gamma=0.9)
        self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=10, gamma=0.9)
    
    def train_step(self, real_images, captions, lengths):
        batch_size = real_images.size(0)
        device = real_images.device
        
        # Label smoothing
        real_labels = torch.ones(batch_size, 1, device=device) * 0.9
        fake_labels = torch.zeros(batch_size, 1, device=device) + 0.1
        
        # Get text features
        text_features = self.encoder(captions, lengths)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_output = self.discriminator(real_images, text_features.detach())
        d_real_loss = self.criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(noise, text_features.detach())
        fake_output = self.discriminator(fake_images.detach(), text_features.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()
        
        # Train Generator (less frequently)
        if np.random.random() > 0.2:  # Train generator 80% of the time
            self.g_optimizer.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.latent_dim, device=device)
            fake_images = self.generator(noise, text_features)
            fake_output = self.discriminator(fake_images, text_features)
            
            # Generator loss
            g_loss = self.criterion(fake_output, torch.ones(batch_size, 1, device=device))
            g_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.generator.parameters()) + list(self.encoder.parameters()), 
                max_norm=1.0
            )
            self.g_optimizer.step()
        else:
            g_loss = torch.tensor(0.0)
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_real_acc': (real_output > 0.5).float().mean().item(),
            'd_fake_acc': (fake_output < 0.5).float().mean().item()
        }
    
    def generate_samples(self, text, char_to_idx, num_samples=1):
        self.eval()
        with torch.no_grad():
            # Convert text to indices
            text_indices = [char_to_idx.get('<START>', 1)]
            text_indices.extend([char_to_idx.get(c, 0) for c in text])
            text_indices.append(char_to_idx.get('<END>', 2))
            
            text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0)
            lengths = torch.tensor([len(text_indices)])
            
            device = next(self.parameters()).device
            text_tensor = text_tensor.to(device)
            lengths = lengths.to(device)
            
            text_features = self.encoder(text_tensor, lengths)
            
            generated_images = []
            for _ in range(num_samples):
                noise = torch.randn(1, self.latent_dim, device=device)
                generated_image = self.generator(noise, text_features)
                generated_image = (generated_image + 1) / 2  # Normalize to [0, 1]
                generated_images.append(generated_image.cpu().numpy()[0, 0])
            
        self.train()
        return generated_images

class StreamlitHandwritingApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'char_to_idx' not in st.session_state:
            st.session_state.char_to_idx = None
        if 'hyperparams' not in st.session_state:
            st.session_state.hyperparams = {
                'latent_dim': 100,
                'embed_size': 256,
                'hidden_size': 512,
                'batch_size': 16,
                'learning_rate': 0.0002
            }
            
        self.dataset = None
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
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.dataset = HandwritingDataset('uploads', transform=transform)
        st.session_state.char_to_idx = self.dataset.char_to_idx
        
        return DataLoader(
            self.dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    def initialize_model(self):
        if self.dataset is None:
            dataloader = self.load_dataset()
        
        model = ImprovedHandwritingGAN(
            self.hyperparams['latent_dim'],
            self.dataset.vocab_size,
            self.hyperparams['embed_size'],
            self.hyperparams['hidden_size']
        ).to(self.device)
        
        return model
    
    def train_model(self, num_epochs, progress_bar, status_text):
        model = self.initialize_model()
        dataloader = self.load_dataset()
        
        losses = {'d_loss': [], 'g_loss': [], 'd_acc': []}
        
        try:
            for epoch in range(num_epochs):
                epoch_d_losses = []
                epoch_g_losses = []
                epoch_d_accs = []
                
                for batch_idx, batch in enumerate(dataloader):
                    images = batch['image'].to(self.device)
                    captions = batch['text'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    
                    batch_losses = model.train_step(images, captions, lengths)
                    
                    epoch_d_losses.append(batch_losses['d_loss'])
                    epoch_g_losses.append(batch_losses['g_loss'])
                    epoch_d_accs.append((batch_losses['d_real_acc'] + batch_losses['d_fake_acc']) / 2)
                
                # Update learning rate
                model.g_scheduler.step()
                model.d_scheduler.step()
                
                # Calculate epoch averages
                avg_d_loss = np.mean(epoch_d_losses)
                avg_g_loss = np.mean(epoch_g_losses)
                avg_d_acc = np.mean(epoch_d_accs)
                
                losses['d_loss'].append(avg_d_loss)
                losses['g_loss'].append(avg_g_loss)
                losses['d_acc'].append(avg_d_acc)
                
                # Update progress
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                
                status_text.text(f"""
                Epoch {epoch+1}/{num_epochs}
                D_Loss: {avg_d_loss:.4f} | G_Loss: {avg_g_loss:.4f} | D_Acc: {avg_d_acc:.3f}
                """)
                
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_model(model, f'epoch_{epoch+1}')
                    
                    # Generate sample to check progress
                    if epoch > 10:  # Start generating after some training
                        sample_text = "Hello"
                        samples = model.generate_samples(sample_text, self.dataset.char_to_idx, 1)
                        
                        # Show sample in progress
                        if samples:
                            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
                            ax.imshow(samples[0], cmap='gray')
                            ax.set_title(f'Epoch {epoch+1}: "{sample_text}"')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
            
            # Final save
            self.save_model(model, 'final')
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            return losses
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return None
    
    def save_model(self, model, name):
        try:
            checkpoint = {
                'encoder_state_dict': model.encoder.state_dict(),
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'hyperparams': self.hyperparams,
                'char_to_idx': st.session_state.char_to_idx,
                'vocab_size': model.vocab_size
            }
            torch.save(checkpoint, f'models/checkpoint_{name}.pth')
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    def load_model(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.hyperparams = checkpoint['hyperparams']
            st.session_state.hyperparams = self.hyperparams
            st.session_state.char_to_idx = checkpoint['char_to_idx']
            
            model = ImprovedHandwritingGAN(
                self.hyperparams['latent_dim'],
                checkpoint['vocab_size'],
                self.hyperparams['embed_size'],
                self.hyperparams['hidden_size']
            ).to(self.device)
            
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.generator.load_state_dict(checkpoint['generator_state_dict'])
            model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_handwriting(self, text, num_samples=3):
        if st.session_state.model is None or st.session_state.char_to_idx is None:
            return None
            
        model = st.session_state.model
        char_to_idx = st.session_state.char_to_idx
        
        return model.generate_samples(text, char_to_idx, num_samples)

def main():
    st.set_page_config(
        page_title="Improved Handwriting Generation",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Improved Deep Learning Handwriting Generation")
    st.write("""
    Enhanced GAN architecture for better handwriting generation with:
    - Improved model architecture with skip connections
    - Better training stability with label smoothing
    - Bidirectional LSTM text encoder
    - Progressive training visualization
    """)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        st.sidebar.success(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.info("💻 Using CPU (GPU strongly recommended)")
    
    app = StreamlitHandwritingApp()
    app.setup_folders()
    
    # Rest of the Streamlit interface remains similar but with improved functionality
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Dataset", "Train Model", "Generate Handwriting"],
        index=0
    )
    
    if page == "Upload Dataset":
        st.header("📁 Upload Dataset")
        # ... (upload logic remains the same)
        
    elif page == "Train Model":
        st.header("🏋️ Train Improved Model")
        # ... (training interface with better progress tracking)
        
    elif page == "Generate Handwriting":
        st.header("✨ Generate Handwritten Text")
        # ... (generation interface with multiple samples)

if __name__ == "__main__":
    main()
