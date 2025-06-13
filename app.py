import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import json
import zipfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Import the model classes from previous code
# (Include all the model classes from the previous code here: HandwritingDataset, Encoder, Attention, Generator, Discriminator, HandwritingGAN)

class StreamlitHandwritingApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset = None
        
        # Model hyperparameters
        self.hyperparams = {
            'latent_dim': 100,
            'vocab_size': 128,
            'embed_size': 256,
            'hidden_size': 512,
            'attention_dim': 256,
            'batch_size': 32,
            'learning_rate': 0.0002
        }
    
    def setup_folders(self):
        """Create necessary folders for the app."""
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('generated', exist_ok=True)
    
    def process_uploaded_dataset(self, uploaded_file):
        """Process uploaded ZIP file containing dataset."""
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
    
    def load_dataset(self):
        """Load the dataset and create DataLoader."""
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
            num_workers=2
        )
    
    def initialize_model(self):
        """Initialize the HandwritingGAN model."""
        self.model = HandwritingGAN(
            self.hyperparams['latent_dim'],
            self.hyperparams['vocab_size'],
            self.hyperparams['embed_size'],
            self.hyperparams['hidden_size'],
            self.hyperparams['attention_dim']
        ).to(self.device)
    
    def train_model(self, num_epochs, progress_bar):
        """Train the model with progress updates."""
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            dataloader = self.load_dataset()
            
            for batch in dataloader:
                images = batch['image'].to(self.device)
                captions = batch['text'].to(self.device)
                lengths = batch['length']
                
                batch_losses = self.model.train_step(images, captions, lengths)
                epoch_losses.append(batch_losses)
                
                progress_text = f"Epoch {epoch+1}/{num_epochs} - "
                progress_text += f"D_Loss: {batch_losses['d_loss']:.4f}, "
                progress_text += f"G_Loss: {batch_losses['g_loss']:.4f}"
                progress_bar.progress((epoch + 1) / num_epochs)
                st.text(progress_text)
            
            if (epoch + 1) % 5 == 0:
                self.save_model(f'epoch_{epoch+1}')
            
            losses.append(np.mean([loss['g_loss'] for loss in epoch_losses]))
        
        return losses
    
    def save_model(self, name):
        """Save model checkpoint."""
        checkpoint = {
            'encoder_state_dict': self.model.encoder.state_dict(),
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'hyperparams': self.hyperparams
        }
        torch.save(checkpoint, f'models/checkpoint_{name}.pth')
    
    def load_model(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.hyperparams = checkpoint['hyperparams']
        
        self.initialize_model()
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    def generate_handwriting(self, text):
        """Generate handwritten text."""
        return self.model.generate_samples(text)

def main():
    st.title("Deep Learning Handwriting Generation")
    st.write("""
    This app uses a GAN (Generative Adversarial Network) to generate handwritten text.
    Upload your dataset, train the model, and generate handwritten text!
    """)
    
    app = StreamlitHandwritingApp()
    app.setup_folders()
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Dataset", "Train Model", "Generate Handwriting"]
    )
    
    if page == "Upload Dataset":
        st.header("Upload Dataset")
        st.write("""
        Upload a ZIP file containing:
        1. Images of handwritten text
        2. labels.json file mapping image filenames to text
        """)
        
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
        if uploaded_file is not None:
            if app.process_uploaded_dataset(uploaded_file):
                st.success("Dataset uploaded and processed successfully!")
                st.write("Found {} samples".format(len(app.load_dataset())))
    
    elif page == "Train Model":
        st.header("Train Model")
        
        if not os.path.exists('uploads/labels.json'):
            st.error("Please upload a dataset first!")
            return
        
        num_epochs = st.slider("Number of epochs", 1, 100, 20)
        
        if app.model is None:
            app.initialize_model()
        
        if st.button("Start Training"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            losses = app.train_model(num_epochs, progress_bar)
            
            fig, ax = plt.subplots()
            ax.plot(losses)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Generator Loss')
            ax.set_title('Training Progress')
            st.pyplot(fig)
            
            st.success("Training completed!")
    
    elif page == "Generate Handwriting":
        st.header("Generate Handwritten Text")
        
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if not model_files:
            st.error("No trained models found! Please train a model first.")
            return
        
        selected_model = st.selectbox("Select a model", model_files)
        
        if app.model is None or selected_model:
            app.load_model(f'models/{selected_model}')
        
        text = st.text_input("Enter text to generate", "Hello World!")
        
        if st.button("Generate"):
            with st.spinner("Generating handwriting..."):
                generated_image = app.generate_handwriting(text)
                
                plt.figure(figsize=(10, 4))
                plt.imshow(generated_image.squeeze(), cmap='gray')
                plt.axis('off')
                st.pyplot(plt)
                
                st.download_button(
                    "Download Image",
                    data=Image.fromarray(generated_image),
                    file_name="generated_handwriting.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
