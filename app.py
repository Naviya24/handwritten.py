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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_path = os.path.join(root_dir, "labels.json")
        
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)
        
        self.image_files = list(self.labels.keys())
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = self.labels[img_name]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'text': [ord(c) for c in label],
            'length': len(label)
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    max_len = max(lengths)
    padded_texts = []
    
    for text in texts:
        padded = text + [0] * (max_len - len(text))
        padded_texts.append(padded)
    
    padded_texts = torch.tensor(padded_texts, dtype=torch.long)
    
    return {
        'image': images,
        'text': padded_texts,
        'length': lengths
    }

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        return hidden[-1]

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.context = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, hidden_states):
        att_weights = torch.tanh(self.attention(hidden_states))
        att_weights = self.context(att_weights).squeeze(-1)
        att_weights = F.softmax(att_weights, dim=1)
        context = torch.sum(att_weights.unsqueeze(-1) * hidden_states, dim=1)
        return context, att_weights

class Generator(nn.Module):
    def __init__(self, latent_dim, text_embed_size, img_size=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + text_embed_size, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, noise, text_embed):
        x = torch.cat([noise, text_embed], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, text_embed_size, img_size=64):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 + text_embed_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, text_embed):
        conv_out = self.conv(img)
        conv_out = conv_out.view(conv_out.size(0), -1)
        x = torch.cat([conv_out, text_embed], dim=1)
        x = self.fc(x)
        return x

class HandwritingGAN(nn.Module):
    def __init__(self, latent_dim, vocab_size, embed_size, hidden_size, attention_dim):
        super(HandwritingGAN, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = TextEncoder(vocab_size, embed_size, hidden_size)
        self.attention = Attention(hidden_size, attention_dim)
        self.generator = Generator(latent_dim, hidden_size)
        self.discriminator = Discriminator(hidden_size)
        
        self.g_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.encoder.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002, betas=(0.5, 0.999)
        )
        
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_images, captions, lengths):
        batch_size = real_images.size(0)
        device = real_images.device
        
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        text_features = self.encoder(captions, lengths)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        real_output = self.discriminator(real_images, text_features.detach())
        d_real_loss = self.criterion(real_output, real_labels)
        
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(noise, text_features.detach())
        fake_output = self.discriminator(fake_images.detach(), text_features.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        fake_output = self.discriminator(fake_images, text_features)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item()
        }
    
    def generate_samples(self, text):
        self.eval()
        with torch.no_grad():
            text_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)
            lengths = [len(text)]
            
            if next(self.parameters()).is_cuda:
                text_tensor = text_tensor.cuda()
            
            text_features = self.encoder(text_tensor, lengths)
            noise = torch.randn(1, self.latent_dim, device=text_features.device)
            
            generated_image = self.generator(noise, text_features)
            generated_image = (generated_image + 1) / 2
            generated_image = generated_image.cpu().numpy()[0, 0]
            
        self.train()
        return generated_image

class StreamlitHandwritingApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize session state for model persistence
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'hyperparams' not in st.session_state:
            st.session_state.hyperparams = {
                'latent_dim': 100,
                'vocab_size': 128,
                'embed_size': 256,
                'hidden_size': 512,
                'attention_dim': 256,
                'batch_size': 32,
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
        return DataLoader(
            self.dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    def initialize_model(self):
        model = HandwritingGAN(
            self.hyperparams['latent_dim'],
            self.hyperparams['vocab_size'],
            self.hyperparams['embed_size'],
            self.hyperparams['hidden_size'],
            self.hyperparams['attention_dim']
        ).to(self.device)
        return model
    
    def train_model(self, num_epochs, progress_bar):
        losses = []
        model = self.initialize_model()
        
        try:
            for epoch in range(num_epochs):
                epoch_losses = []
                dataloader = self.load_dataset()
                
                for batch in dataloader:
                    images = batch['image'].to(self.device)
                    captions = batch['text'].to(self.device)
                    lengths = batch['length']
                    
                    batch_losses = model.train_step(images, captions, lengths)
                    epoch_losses.append(batch_losses)
                    
                progress_text = f"Epoch {epoch+1}/{num_epochs} - "
                progress_text += f"D_Loss: {np.mean([loss['d_loss'] for loss in epoch_losses]):.4f}, "
                progress_text += f"G_Loss: {np.mean([loss['g_loss'] for loss in epoch_losses]):.4f}"
                progress_bar.progress((epoch + 1) / num_epochs)
                st.text(progress_text)
                
                if (epoch + 1) % 5 == 0:
                    self.save_model(model, f'epoch_{epoch+1}')
                
                losses.append(np.mean([loss['g_loss'] for loss in epoch_losses]))
            
            return losses
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return []
    
    def save_model(self, model, name):
        try:
            checkpoint = {
                'encoder_state_dict': model.encoder.state_dict(),
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'hyperparams': self.hyperparams
            }
            torch.save(checkpoint, f'models/checkpoint_{name}.pth')
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    def load_model(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.hyperparams = checkpoint['hyperparams']
            st.session_state.hyperparams = self.hyperparams
            
            # Initialize and load model
            model = HandwritingGAN(
                self.hyperparams['latent_dim'],
                self.hyperparams['vocab_size'],
                self.hyperparams['embed_size'],
                self.hyperparams['hidden_size'],
                self.hyperparams['attention_dim']
            ).to(self.device)
            
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
    
    def generate_handwriting(self, text):
        # Use model from session state
        if st.session_state.model is None:
            return None
            
        model = st.session_state.model
        model.eval()
        
        with torch.no_grad():
            text_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)
            lengths = [len(text)]
            
            if next(model.parameters()).is_cuda:
                text_tensor = text_tensor.cuda()
            
            text_features = model.encoder(text_tensor, lengths)
            noise = torch.randn(1, model.latent_dim, device=text_features.device)
            
            generated_image = model.generator(noise, text_features)
            generated_image = (generated_image + 1) / 2
            generated_image = generated_image.cpu().numpy()[0, 0]
            
        return generated_image

def main():
    st.set_page_config(
        page_title="Deep Learning Handwriting Generation",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Deep Learning Handwriting Generation")
    st.write("""
    This app uses a GAN (Generative Adversarial Network) to generate handwritten text.
    Upload your dataset, train the model, and generate handwritten text!
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
        1. **Images** of handwritten text (PNG, JPG, etc.)
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
  "image3.png": "Deep Learning with PyTorch"
}
            ''', language='json')
        
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
        if uploaded_file is not None:
            if app.process_uploaded_dataset(uploaded_file):
                st.success("✅ Dataset uploaded and processed successfully!")
                try:
                    dataloader = app.load_dataset()
                    st.write(f"📊 Found **{len(dataloader.dataset)}** samples")
                    
                    # Show sample data
                    sample_batch = next(iter(dataloader))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Batch Size", sample_batch['image'].shape[0])
                        st.metric("Image Shape", f"{sample_batch['image'].shape[2]}×{sample_batch['image'].shape[3]}")
                    with col2:
                        st.metric("Total Batches", len(dataloader))
                        st.metric("Max Text Length", max(sample_batch['length']))
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
    
    elif page == "Train Model":
        st.header("🏋️ Train Model")
        
        if not os.path.exists('uploads/labels.json'):
            st.error("❌ Please upload a dataset first!")
            st.info("👈 Go to 'Upload Dataset' page to upload your training data")
            return
        
        # Show dataset info
        try:
            with open('uploads/labels.json', 'r') as f:
                labels = json.load(f)
            st.success(f"✅ Dataset ready: {len(labels)} samples")
        except:
            st.error("❌ Invalid dataset format")
            return
        
        # Hyperparameter controls
        st.subheader("⚙️ Training Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.slider("🔄 Number of epochs", 1, 100, 20, 
                                 help="More epochs = better quality but longer training")
            batch_size = st.selectbox("📦 Batch size", [8, 16, 32, 64], index=2,
                                    help="Larger batch = faster training but more memory")
            learning_rate = st.selectbox("📈 Learning rate", [0.0001, 0.0002, 0.0005], index=1)
        
        with col2:
            latent_dim = st.selectbox("🎲 Latent dimension", [50, 100, 200], index=1,
                                    help="Higher = more complex generations")
            hidden_size = st.selectbox("🧠 Hidden size", [256, 512, 1024], index=1,
                                     help="Larger = more model capacity")
            embed_size = st.selectbox("📝 Embedding size", [128, 256, 512], index=1)
        
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
        estimated_time = (num_epochs * len(labels) // batch_size) * (2 if device.type == 'cpu' else 0.5)
        st.info(f"⏱️ Estimated training time: ~{estimated_time:.1f} minutes")
        
        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔥 Training in progress..."):
                losses = app.train_model(num_epochs, progress_bar)
            
            if losses:
                st.success("🎉 Training completed successfully!")
                
                # Plot training progress
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(losses, 'b-', linewidth=2, marker='o', markersize=4)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Generator Loss')
                ax.set_title('Training Progress')
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8f9fa')
                st.pyplot(fig)
                
                st.balloons()
                st.info("💾 Models saved automatically every 5 epochs in the 'models' folder")
    
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
            selected_model = st.selectbox("🤖 Select a model", model_files,
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
                    st.rerun()
        
        # Text input and generation
        st.subheader("📝 Enter Text to Generate")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_input("Text input", "Hello World!", 
                               help="Enter the text you want to convert to handwriting")
        with col2:
            st.metric("Text Length", len(text))
        
        # Generation tips
        with st.expander("💡 Generation Tips"):
            st.write("""
            - **Start simple**: Try single words like "Hello", "Test", "AI"
            - **Character support**: ASCII characters work best (a-z, A-Z, 0-9, basic punctuation)
            - **Length**: Shorter text (5-15 characters) usually works better
            - **Quality**: Model quality depends on training epochs and dataset
            """)
        
        if st.button("🎨 Generate Handwriting", type="primary"):
            if not st.session_state.model_loaded or st.session_state.model is None:
                st.error("❌ Please load a model first!")
                return
            
            if not text.strip():
                st.error("❌ Please enter some text to generate")
                return
                
            with st.spinner("🎨 Generating handwriting..."):
                try:
                    generated_image = app.generate_handwriting(text)
                    
                    if generated_image is not None:
                        # Display generated image
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.imshow(generated_image, cmap='gray', interpolation='bilinear')
                        ax.set_title(f'Generated Handwriting: "{text}"', fontsize=16, pad=20)
                        ax.axis('off')
                        ax.set_facecolor('white')
                        fig.patch.set_facecolor('white')
                        st.pyplot(fig)
                        
                        # Download button
                        img_array = (generated_image * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_array)
                        
                        buf = io.BytesIO()
                        img_pil.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            "💾 Download Generated Image",
                            data=byte_im,
                            file_name=f"handwriting_{text.replace(' ', '_')}.png",
                            mime="image/png"
                        )
                        
                    else:
                        st.error("❌ Failed to generate image")
                        
                except Exception as e:
                    st.error(f"❌ Error generating handwriting: {str(e)}")
    
    # Sidebar info
    with st.sidebar:
        st.write("---")
        st.subheader("📊 App Info")
        st.write(f"**Device**: {device.type.upper()}")
        if st.session_state.model_loaded:
            st.write("**Model Status**: ✅ Loaded")
        else:
            st.write("**Model Status**: ❌ Not Loaded")
        
        st.write("---")
        st.subheader("🔗 Quick Links")
        st.write("• [PyTorch Documentation](https://pytorch.org/docs/)")
        st.write("• [Streamlit Documentation](https://docs.streamlit.io/)")

if __name__ == "__main__":
    main()
