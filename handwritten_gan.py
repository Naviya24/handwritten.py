import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import random
import cv2
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time

# Set page config
st.set_page_config(
    page_title="AI-Powered Handwriting Synthesis Lab",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Generator with Attention Mechanism
class AttentionBlock(nn.Module):
    def __init__(self, in_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(in_dim, in_dim // 4)
        self.key = nn.Linear(in_dim, in_dim // 4)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)))
        attended = torch.matmul(attention_weights, V)
        return attended + x  # Residual connection

class AdvancedGenerator(nn.Module):
    def __init__(self, text_dim=256, noise_dim=100, style_dim=50, img_dim=784):
        super(AdvancedGenerator, self).__init__()
        self.text_dim = text_dim
        self.noise_dim = noise_dim
        self.style_dim = style_dim
        
        # Advanced text embedding with transformer-like architecture
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            AttentionBlock(512),
            nn.Linear(512, 256)
        )
        
        # Style conditioning network
        self.style_embedding = nn.Sequential(
            nn.Linear(style_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Noise processing with residual blocks
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Multi-scale generation network
        self.generator_blocks = nn.ModuleList([
            self._make_generator_block(768, 1024),
            self._make_generator_block(1024, 2048),
            self._make_generator_block(2048, 4096),
        ])
        
        # Final output layers for different resolutions
        self.output_28x28 = nn.Linear(4096, 784)
        self.output_56x56 = nn.Linear(4096, 3136)
        self.output_112x112 = nn.Linear(4096, 12544)
        
        self.final_activation = nn.Tanh()
        
    def _make_generator_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
    def forward(self, text_features, noise, style_vector, output_size=28):
        text_emb = self.text_embedding(text_features)
        style_emb = self.style_embedding(style_vector)
        noise_emb = self.noise_fc(noise)
        
        # Concatenate all embeddings
        combined = torch.cat([text_emb, style_emb, noise_emb], dim=1)
        
        # Pass through generator blocks
        x = combined
        for block in self.generator_blocks:
            x = block(x)
        
        # Generate different resolutions
        if output_size == 28:
            output = self.output_28x28(x)
        elif output_size == 56:
            output = self.output_56x56(x)
        elif output_size == 112:
            output = self.output_112x112(x)
        else:
            output = self.output_28x28(x)
            
        return self.final_activation(output)

# Advanced Text Encoder with Linguistic Features
class LinguisticTextEncoder:
    def __init__(self):
        self.char_to_idx = {chr(i): i-32 for i in range(32, 127)}
        self.char_to_idx[' '] = 0
        
        # Linguistic feature extractors
        self.vowels = set('aeiouAEIOU')
        self.consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
    def extract_linguistic_features(self, text):
        features = {}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character composition
        features['vowel_ratio'] = sum(1 for c in text if c in self.vowels) / len(text) if text else 0
        features['consonant_ratio'] = sum(1 for c in text if c in self.consonants) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['upper_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['space_ratio'] = sum(1 for c in text if c.isspace()) / len(text) if text else 0
        
        # Complexity measures
        features['unique_chars'] = len(set(text.lower()))
        features['entropy'] = self._calculate_entropy(text)
        features['readability'] = self._flesch_reading_ease(text)
        
        return features
    
    def _calculate_entropy(self, text):
        if not text:
            return 0
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        entropy = 0
        for freq in char_freq.values():
            p = freq / len(text)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _flesch_reading_ease(self, text):
        if not text:
            return 0
        sentences = text.count('.') + text.count('!') + text.count('?') + 1
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if words == 0 or sentences == 0:
            return 0
            
        return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    
    def _count_syllables(self, word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
            
        if word.endswith('e'):
            syllable_count -= 1
        if syllable_count == 0:
            syllable_count = 1
            
        return syllable_count
    
    def encode_text_advanced(self, text, max_len=100):
        # Character-level encoding
        char_features = np.zeros(max_len)
        for i, char in enumerate(text[:max_len]):
            if char in self.char_to_idx:
                char_features[i] = self.char_to_idx[char] / 95.0
        
        # Linguistic features
        ling_features = self.extract_linguistic_features(text)
        ling_vector = np.array(list(ling_features.values()))
        
        # Combine features
        combined_features = np.concatenate([
            char_features[:200],  # First 200 char features
            ling_vector[:56]      # Linguistic features to make total 256
        ])
        
        # Pad to exactly 256 dimensions
        if len(combined_features) < 256:
            combined_features = np.pad(combined_features, (0, 256 - len(combined_features)))
        
        return combined_features[:256]

# Style Vector Generator
class HandwritingStyleAnalyzer:
    def __init__(self):
        self.style_parameters = {
            'casual': {
                'slant': 0.1, 'thickness': 1.0, 'spacing': 1.0, 'irregularity': 0.15,
                'pressure_variation': 0.2, 'speed': 0.7, 'fluidity': 0.6
            },
            'formal': {
                'slant': 0.05, 'thickness': 0.8, 'spacing': 1.2, 'irregularity': 0.05,
                'pressure_variation': 0.1, 'speed': 0.4, 'fluidity': 0.3
            },
            'cursive': {
                'slant': 0.2, 'thickness': 0.9, 'spacing': 0.8, 'irregularity': 0.12,
                'pressure_variation': 0.25, 'speed': 0.8, 'fluidity': 0.9
            },
            'bold': {
                'slant': 0.0, 'thickness': 1.5, 'spacing': 1.1, 'irregularity': 0.08,
                'pressure_variation': 0.3, 'speed': 0.5, 'fluidity': 0.4
            },
            'elegant': {
                'slant': 0.15, 'thickness': 0.7, 'spacing': 1.3, 'irregularity': 0.06,
                'pressure_variation': 0.15, 'speed': 0.3, 'fluidity': 0.8
            },
            'artistic': {
                'slant': 0.25, 'thickness': 1.2, 'spacing': 0.9, 'irregularity': 0.3,
                'pressure_variation': 0.4, 'speed': 0.9, 'fluidity': 0.7
            }
        }
    
    def get_style_vector(self, style_name, randomize=False):
        if style_name not in self.style_parameters:
            style_name = 'casual'
        
        params = self.style_parameters[style_name].copy()
        
        if randomize:
            for key in params:
                params[key] += np.random.normal(0, 0.1)
                params[key] = np.clip(params[key], 0, 2)
        
        # Convert to vector format (50 dimensions)
        base_vector = np.array(list(params.values()))
        
        # Add derived features
        additional_features = [
            params['slant'] * params['thickness'],
            params['spacing'] / (params['irregularity'] + 0.01),
            params['pressure_variation'] * params['speed'],
            params['fluidity'] * (1 - params['irregularity']),
            np.sqrt(params['thickness'] * params['pressure_variation']),
        ]
        
        extended_vector = np.concatenate([base_vector, additional_features])
        
        # Pad to 50 dimensions
        if len(extended_vector) < 50:
            extended_vector = np.pad(extended_vector, (0, 50 - len(extended_vector)))
        
        return extended_vector[:50]

# Advanced Image Processing
class HandwritingPostProcessor:
    def __init__(self):
        self.filters = {
            'blur': ImageFilter.GaussianBlur(radius=0.5),
            'sharpen': ImageFilter.SHARPEN,
            'edge_enhance': ImageFilter.EDGE_ENHANCE,
        }
    
    def apply_realistic_effects(self, image, style_params):
        """Apply realistic handwriting effects (softened for visibility)"""
        # Convert to numpy for processing
        img_array = np.array(image)

        # Add paper texture (reduced noise)
        img_array = np.clip(img_array + np.random.normal(0, 2, img_array.shape), 0, 255)

        # Simulate ink bleeding (minimal blur)
        thickness = style_params.get('thickness', 1.0)
        kernel_size = 1  # Minimal blur
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        if len(img_array.shape) == 3:
         for i in range(img_array.shape[2]):
            img_array[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel)
        else:
         img_array = cv2.filter2D(img_array, -1, kernel)

        # Add minimal pressure variation
        variation = style_params.get('pressure_variation', 0.1)
        pressure_map = np.ones(img_array.shape[:2]) * (1 - variation * 0.1)

        if len(img_array.shape) == 3:
         for i in range(img_array.shape[2]):
            img_array[:, :, i] = img_array[:, :, i] * pressure_map
        else:
         img_array = img_array * pressure_map

        img_array = np.clip(img_array, 0, 255)

        # Add slight yellowing (aging)
        if len(img_array.shape) == 3:
          img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.01, 0, 255)  # Red
          img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.01, 0, 255)  # Green

        # Increase contrast to enhance visibility
        output_image = Image.fromarray(img_array.astype(np.uint8))
        output_image = ImageEnhance.Contrast(output_image).enhance(1.5)
        return output_image

    
    def _add_paper_texture(self, img_array):
        """Add subtle paper texture"""
        texture = np.random.normal(0, 5, img_array.shape)
        return np.clip(img_array + texture, 0, 255)
    
    def _simulate_ink_bleeding(self, img_array, thickness):
        """Simulate ink bleeding effect"""
        kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        if len(img_array.shape) == 3:
            for i in range(img_array.shape[2]):
                img_array[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel)
        else:
            img_array = cv2.filter2D(img_array, -1, kernel)
        
        return img_array
    
    def _add_pressure_variations(self, img_array, variation):
        """Add pressure variation effects"""
        # Create pressure map
        pressure_map = np.ones(img_array.shape[:2]) * 0.1  # minimal variation

        pressure_map = ndimage.gaussian_filter(pressure_map, sigma=5)
        
        # Apply pressure variations
        if len(img_array.shape) == 3:
            for i in range(img_array.shape[2]):
                img_array[:, :, i] = img_array[:, :, i] * (1 - variation * pressure_map)
        else:
            img_array = img_array * (1 - variation * pressure_map)
        
        return np.clip(img_array, 0, 255)
    
    def _add_aging_effects(self, img_array):
        """Add subtle aging effects"""
        # Add slight yellowing
        if len(img_array.shape) == 3:
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.02, 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.01, 0, 255)  # Green
        
        return img_array

# Analytics and Visualization
class HandwritingAnalytics:
    def __init__(self):
        self.generation_history = []
    
    def add_generation(self, text, style, method, quality_score):
        self.generation_history.append({
            'timestamp': datetime.now(),
            'text': text,
            'style': style,
            'method': method,
            'quality_score': quality_score,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    def get_analytics_dashboard(self):
        if not self.generation_history:
            return None
        
        df = pd.DataFrame(self.generation_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Style Distribution', 'Quality Scores Over Time', 
                          'Text Length Distribution', 'Method Comparison'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Style distribution pie chart
        style_counts = df['style'].value_counts()
        fig.add_trace(
            go.Pie(labels=style_counts.index, values=style_counts.values, name="Styles"),
            row=1, col=1
        )
        
        # Quality scores over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['quality_score'], 
                      mode='lines+markers', name="Quality"),
            row=1, col=2
        )
        
        # Text length distribution
        fig.add_trace(
            go.Histogram(x=df['text_length'], name="Length"),
            row=2, col=1
        )
        
        # Method comparison
        method_quality = df.groupby('method')['quality_score'].mean()
        fig.add_trace(
            go.Bar(x=method_quality.index, y=method_quality.values, name="Avg Quality"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, 
                         title_text="Handwriting Generation Analytics")
        return fig

# Quality Assessment
class QualityAssessment:
    def __init__(self):
        self.metrics = {}
    
    def assess_image_quality(self, image):
        """Assess the quality of generated handwriting"""
        img_array = np.array(image.convert('L'))
        
        metrics = {}
        
        # Sharpness (Laplacian variance)
        metrics['sharpness'] = cv2.Laplacian(img_array, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        metrics['contrast'] = np.std(img_array)
        
        # Edge density
        edges = cv2.Canny(img_array, 50, 150)
        metrics['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Noise level (high frequency content)
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        high_freq = magnitude[magnitude.shape[0]//4:3*magnitude.shape[0]//4, 
                             magnitude.shape[1]//4:3*magnitude.shape[1]//4]
        metrics['noise_level'] = np.mean(high_freq)
        
        # Overall quality score (normalized)
        quality_score = (
            min(metrics['sharpness'] / 1000, 1) * 0.3 +
            min(metrics['contrast'] / 100, 1) * 0.3 +
            min(metrics['edge_density'] * 10, 1) * 0.2 +
            max(0, 1 - metrics['noise_level'] / 10000) * 0.2
        )
        
        return quality_score, metrics

# Load models and components
@st.cache_resource
def load_advanced_models():
    generator = AdvancedGenerator()
    generator.eval()
    text_encoder = LinguisticTextEncoder()
    style_analyzer = HandwritingStyleAnalyzer()
    post_processor = HandwritingPostProcessor()
    analytics = HandwritingAnalytics()
    quality_assessor = QualityAssessment()
    
    return generator, text_encoder, style_analyzer, post_processor, analytics, quality_assessor

# Advanced generation function
def generate_advanced_handwriting(text, style, output_size, generator, text_encoder, 
                                style_analyzer, post_processor, seed=None, 
                                custom_params=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Encode text with linguistic features
    text_features = text_encoder.encode_text_advanced(text)
    text_tensor = torch.FloatTensor(text_features).unsqueeze(0)
    
    # Get style vector
    style_vector = style_analyzer.get_style_vector(style, randomize=True)
    style_tensor = torch.FloatTensor(style_vector).unsqueeze(0)
    
    # Generate noise
    noise = torch.randn(1, 100)
    
    # Generate image
    with torch.no_grad():
        generated = generator(text_tensor, noise, style_tensor, output_size)
        
        if output_size == 28:
            generated = generated.view(28, 28).numpy()
        elif output_size == 56:
            generated = generated.view(56, 56).numpy()
        elif output_size == 112:
            generated = generated.view(112, 112).numpy()
    
    # Convert to PIL image
    generated = (generated + 1) / 2
    generated = (generated * 255).astype(np.uint8)
    img = Image.fromarray(generated, mode='L')
    
    # Resize for display
    display_size = max(400, len(text) * 20)
    img = img.resize((display_size, 200), Image.LANCZOS)
    
    # Apply post-processing effects
    style_params = style_analyzer.style_parameters[style]
    img = post_processor.apply_realistic_effects(img, style_params)
    
    return img

# Create realistic handwriting with advanced features
def create_ultra_realistic_handwriting(text, style, custom_params=None):
    """Create ultra-realistic handwriting with advanced effects"""
    
    # Dynamic sizing based on text length
    char_width = 25
    base_width = max(600, len(text) * char_width)
    base_height = 150
    
    # Multi-line support
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    max_line_length = base_width // char_width
    
    for word in words:
        if current_length + len(word) + 1 <= max_line_length:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Adjust height for multiple lines
    final_height = base_height * max(1, len(lines))
    img = Image.new('RGB', (base_width, final_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Style parameters
    style_analyzer = HandwritingStyleAnalyzer()
    style_params = style_analyzer.style_parameters.get(style, 
                   style_analyzer.style_parameters['casual'])
    
    # Font and color variations
    colors = {
        'casual': [(50, 50, 150), (60, 40, 140), (40, 60, 160)],
        'formal': [(0, 0, 100), (10, 10, 90), (0, 10, 110)],
        'cursive': [(80, 50, 120), (90, 40, 110), (70, 60, 130)],
        'bold': [(20, 20, 20), (30, 30, 30), (10, 10, 10)],
        'elegant': [(70, 40, 90), (80, 30, 80), (60, 50, 100)],
        'artistic': [(120, 80, 40), (110, 90, 50), (130, 70, 30)]
    }
    
    style_colors = colors.get(style, colors['casual'])
    
    try:
        font_size = int(40 * style_params['thickness'])
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw each line with advanced effects
    for line_idx, line in enumerate(lines):
        y_base = 50 + line_idx * (base_height - 20)
        x = 50
        
        # Line-level variations
        line_slant = style_params['slant'] * np.random.uniform(-1, 1)
        line_baseline_var = np.random.uniform(-5, 5)
        
        for word_idx, word in enumerate(line.split()):
            # Word-level variations
            word_color = random.choice(style_colors)
            word_spacing = style_params['spacing'] * (1 + np.random.uniform(-0.2, 0.2))
            
            for char_idx, char in enumerate(word):
                # Character-level variations
                char_color = tuple(max(0, min(255, c + np.random.randint(-20, 20))) 
                                 for c in word_color)
                
                # Position variations
                char_x_offset = np.random.uniform(-style_params['irregularity'] * 10, 
                                                style_params['irregularity'] * 10)
                char_y_offset = (np.random.uniform(-style_params['irregularity'] * 8, 
                                                 style_params['irregularity'] * 8) + 
                               line_baseline_var)
                
                # Pressure simulation through transparency
                pressure = style_params['pressure_variation']
                alpha_variation = int(255 * (1 - pressure * np.random.uniform(0, 0.5)))
                
                # Draw character with effects
                char_x = x + char_x_offset + char_idx * line_slant * 2
                char_y = y_base + char_y_offset
                
                # Create temporary image for character with alpha
                char_img = Image.new('RGBA', (50, 60), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((10, 10), char, fill=(*char_color, alpha_variation), font=font)
                
                # Paste with blending
                img.paste(char_img, (int(char_x-10), int(char_y-10)), char_img)
                
                # Update x position
                char_width_actual = draw.textlength(char, font=font) * word_spacing
                x += char_width_actual + np.random.uniform(-2, 2)
            
            # Add word spacing
            x += 15 * word_spacing
    
    return img

# Main Streamlit Application
def main():
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>✍️ AI-Powered Handwriting Synthesis Lab</h1>
        <p style='font-size: 1.2rem; margin: 0;'>Advanced Neural Handwriting Generation with Linguistic Analysis & Style Transfer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    (generator, text_encoder, style_analyzer, post_processor, 
     analytics, quality_assessor) = load_advanced_models()
    
    # Initialize session state for analytics
    if 'analytics' not in st.session_state:
        st.session_state.analytics = analytics
    
    # Sidebar Configuration
    st.sidebar.markdown("##  Generation Controls")
    
    # Text input with preprocessing options
    st.sidebar.markdown("###  Text Input")
    input_text = st.sidebar.text_area(
        "Enter text to convert:",
        value="The future of AI is handwritten!",
        max_chars=500,
        help="Enter text to convert to handwriting (supports multi-line)"
    )
    
    # Text preprocessing options
    preprocess_options = st.sidebar.multiselect(
        "Text Preprocessing:",
        ["Auto-capitalize", "Remove extra spaces", "Add punctuation"],
        default=["Remove extra spaces"]
    )
    
    # Apply preprocessing
    processed_text = input_text
    if "Auto-capitalize" in preprocess_options:
        processed_text = '. '.join(s.strip().capitalize() for s in processed_text.split('.'))
    if "Remove extra spaces" in preprocess_options:
        processed_text = ' '.join(processed_text.split())
    if "Add punctuation" in preprocess_options and processed_text and not processed_text[-1] in '.!?':
        processed_text += '.'
    
    # Style selection with preview
    st.sidebar.markdown("###  Style Configuration")
    styles = ["casual", "formal", "cursive", "bold", "elegant", "artistic"]
    style = st.sidebar.selectbox("Handwriting Style:", styles, index=0)
    
    # Style parameter customization
    if st.sidebar.checkbox(" Advanced Style Tuning"):
        st.sidebar.markdown("#### Custom Style Parameters")
        custom_slant = st.sidebar.slider("Slant", -0.5, 0.5, 0.1, 0.05)
        custom_thickness = st.sidebar.slider("Thickness", 0.5, 2.0, 1.0, 0.1)
        custom_irregularity = st.sidebar.slider("Irregularity", 0.0, 0.5, 0.15, 0.05)
        custom_spacing = st.sidebar.slider("Spacing", 0.5, 2.0, 1.0, 0.1)
        
        custom_params = {
            'slant': custom_slant,
            'thickness': custom_thickness,
            'irregularity': custom_irregularity,
            'spacing': custom_spacing
        }
    else:
        custom_params = None
    
    # Generation method and quality settings
    st.sidebar.markdown("###  Generation Settings")
    method = st.sidebar.radio(
        "Generation Method:",
        ["Ultra-Realistic (Advanced PIL)", "Neural GAN Model", "Hybrid Approach"],
        help="Choose generation approach"
    )
    
    if method == "Neural GAN Model":
        output_size = st.sidebar.selectbox(
            "Output Resolution:",
            [28, 56, 112],
            index=1,
            help="Higher resolution = better quality, slower generation"
        )
    else:
        output_size = 56
    
    # Quality and performance settings
    quality_mode = st.sidebar.selectbox(
        "Quality Mode:",
        ["Standard", "High Quality", "Ultra High"],
        index=1
    )
    
    # Seed and randomization
    st.sidebar.markdown("###  Randomization")
    use_seed = st.sidebar.checkbox("Use seed for reproducible results")
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Seed:", value=42, min_value=0, max_value=100000)
    
    # Batch generation
    batch_mode = st.sidebar.checkbox(" Batch Generation Mode")
    if batch_mode:
        batch_count = st.sidebar.slider("Number of variations:", 2, 10, 3)
    else:
        batch_count = 1
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Text Analysis")
        
        if processed_text.strip():
            # Display linguistic analysis
            ling_features = text_encoder.extract_linguistic_features(processed_text)
            
            # Create analysis visualization
            analysis_cols = st.columns(4)
            
            with analysis_cols[0]:
                st.metric("Characters", ling_features['length'])
                st.metric("Words", ling_features['word_count'])
            
            with analysis_cols[1]:
                st.metric("Avg Word Length", f"{ling_features['avg_word_length']:.1f}")
                st.metric("Unique Characters", ling_features['unique_chars'])
            
            with analysis_cols[2]:
                st.metric("Vowel Ratio", f"{ling_features['vowel_ratio']:.2f}")
                st.metric("Text Entropy", f"{ling_features['entropy']:.2f}")
            
            with analysis_cols[3]:
                st.metric("Readability Score", f"{ling_features['readability']:.1f}")
                st.metric("Complexity", "High" if ling_features['entropy'] > 4 else "Medium" if ling_features['entropy'] > 3 else "Low")
            
            # Text complexity visualization
            complexity_data = {
                'Metric': ['Vowel Ratio', 'Consonant Ratio', 'Digit Ratio', 'Upper Case Ratio'],
                'Value': [ling_features['vowel_ratio'], ling_features['consonant_ratio'], 
                         ling_features['digit_ratio'], ling_features['upper_ratio']]
            }
            
            fig_complexity = px.bar(
                complexity_data, x='Metric', y='Value',
                title="Text Composition Analysis",
                color='Value',
                color_continuous_scale='viridis'
            )
            fig_complexity.update_layout(height=300)
            st.plotly_chart(fig_complexity, use_container_width=True)
    
    with col2:
        st.markdown("##  Style Preview")
        
        # Style parameters visualization
        style_params = style_analyzer.style_parameters[style]
        
        # Create radar chart for style parameters
        categories = list(style_params.keys())
        values = list(style_params.values())
        
        fig_style = go.Figure()
        fig_style.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=style.title(),
            line_color='rgb(32, 201, 151)'
        ))
        
        fig_style.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 2])
            ),
            showlegend=True,
            title=f"{style.title()} Style Profile",
            height=300
        )
        st.plotly_chart(fig_style, use_container_width=True)
    
    # Generation section
    st.markdown("---")
    st.markdown("##  Generate Handwriting")
    
    generation_cols = st.columns([1, 1, 1])
    
    with generation_cols[1]:
        generate_btn = st.button(
            " Generate Handwritten Text",
            type="primary",
            use_container_width=True
        )
    
    if generate_btn and processed_text.strip():
        with st.spinner(" AI is crafting your handwriting..."):
            start_time = time.time()
            
            # Generate multiple variations if batch mode
            generated_images = []
            quality_scores = []
            
            for i in range(batch_count):
                current_seed = seed + i if seed else None
                
                try:
                    if method == "Ultra-Realistic (Advanced PIL)":
                        img = create_ultra_realistic_handwriting(
                            processed_text, style, custom_params
                        )
                    elif method == "Neural GAN Model":
                        img = generate_advanced_handwriting(
                            processed_text, style, output_size, generator,
                            text_encoder, style_analyzer, post_processor,
                            current_seed, custom_params
                        )
                    else:  # Hybrid Approach
                        # Generate with both methods and blend
                        img1 = create_ultra_realistic_handwriting(
                            processed_text, style, custom_params
                        )
                        img2 = generate_advanced_handwriting(
                            processed_text, style, output_size, generator,
                            text_encoder, style_analyzer, post_processor,
                            current_seed, custom_params
                        )
                        # Simple blending (you could make this more sophisticated)
                        img = Image.blend(img1.resize(img2.size), img2, 0.5)
                    
                    generated_images.append(img)
                    
                    # Assess quality
                    quality_score, metrics = quality_assessor.assess_image_quality(img)
                    quality_scores.append(quality_score)
                    
                    # Add to analytics
                    st.session_state.analytics.add_generation(
                        processed_text, style, method, quality_score
                    )
                    
                except Exception as e:
                    st.error(f"Generation failed for variation {i+1}: {str(e)}")
                    continue
            
            generation_time = time.time() - start_time
            
            if generated_images:
                st.success(f"Generated {len(generated_images)} variations in {generation_time:.2f}s")
                
                # Display results
                if batch_count == 1:
                    # Single image display
                    result_cols = st.columns([2, 1])
                    
                    with result_cols[0]:
                        st.markdown("###  Generated Handwriting")
                        st.image(generated_images[0], caption=f"Style: {style.title()}")
                        
                        # Download options
                        download_cols = st.columns(3)
                        
                        with download_cols[0]:
                            buf = io.BytesIO()
                            generated_images[0].save(buf, format='PNG')
                            buf.seek(0)
                            st.download_button(
                                " Download PNG",
                                data=buf.getvalue(),
                                file_name=f"handwriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        
                        with download_cols[1]:
                            buf_jpg = io.BytesIO()
                            generated_images[0].convert('RGB').save(buf_jpg, format='JPEG', quality=95)
                            buf_jpg.seek(0)
                            st.download_button(
                                " Download JPG",
                                data=buf_jpg.getvalue(),
                                file_name=f"handwriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg"
                            )
                        
                        with download_cols[2]:
                            # Save generation metadata
                            metadata = {
                                'text': processed_text,
                                'style': style,
                                'method': method,
                                'quality_score': quality_scores[0],
                                'generation_time': generation_time,
                                'parameters': custom_params,
                                'linguistic_features': ling_features
                            }
                            metadata_json = json.dumps(metadata, indent=2, default=str)
                            st.download_button(
                                " Download Metadata",
                                data=metadata_json,
                                file_name=f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    with result_cols[1]:
                        st.markdown("###  Quality Analysis")
                        
                        # Quality metrics
                        quality_score, metrics = quality_scores[0], quality_assessor.assess_image_quality(generated_images[0])[1]
                        
                        st.metric("Overall Quality", f"{quality_score:.3f}")
                        st.metric("Sharpness", f"{metrics['sharpness']:.0f}")
                        st.metric("Contrast", f"{metrics['contrast']:.1f}")
                        st.metric("Edge Density", f"{metrics['edge_density']:.3f}")
                        
                        # Quality visualization
                        quality_data = {
                            'Metric': ['Sharpness', 'Contrast', 'Edge Density', 'Clarity'],
                            'Score': [
                                min(metrics['sharpness'] / 1000, 1),
                                min(metrics['contrast'] / 100, 1),
                                min(metrics['edge_density'] * 10, 1),
                                max(0, 1 - metrics['noise_level'] / 10000)
                            ]
                        }
                        
                        fig_quality = px.bar(
                            quality_data, x='Metric', y='Score',
                            title="Quality Breakdown",
                            color='Score',
                            color_continuous_scale='RdYlGn'
                        )
                        fig_quality.update_layout(height=300)
                        st.plotly_chart(fig_quality, use_container_width=True)
                
                else:
                    # Batch display
                    st.markdown("###  Generated Variations")
                    
                    batch_cols = st.columns(min(3, batch_count))
                    for i, img in enumerate(generated_images):
                        with batch_cols[i % 3]:
                            st.image(img, caption=f"Variation {i+1} (Quality: {quality_scores[i]:.3f})")
                    
                    # Best variation selector
                    best_idx = np.argmax(quality_scores)
                    st.info(f" Best variation: #{best_idx + 1} with quality score {quality_scores[best_idx]:.3f}")
                
                # Generation insights
                st.markdown("###  Generation Insights")
                insights_cols = st.columns(3)
                
                with insights_cols[0]:
                    st.markdown("**Text Complexity Impact:**")
                    complexity_impact = "High" if ling_features['entropy'] > 4 else "Medium"
                    st.write(f"- Entropy: {ling_features['entropy']:.2f} ({complexity_impact})")
                    st.write(f"- Readability: {ling_features['readability']:.1f}")
                    st.write(f"- Character variety: {ling_features['unique_chars']} unique")
                
                with insights_cols[1]:
                    st.markdown("**Style Characteristics:**")
                    st.write(f"- Dominant feature: {max(style_params, key=style_params.get)}")
                    st.write(f"- Irregularity level: {style_params['irregularity']:.2f}")
                    st.write(f"- Thickness factor: {style_params['thickness']:.2f}")
                
                with insights_cols[2]:
                    st.markdown("**Performance Metrics:**")
                    st.write(f"- Generation time: {generation_time:.2f}s")
                    st.write(f"- Average quality: {np.mean(quality_scores):.3f}")
                    st.write(f"- Method efficiency: {len(processed_text)/generation_time:.1f} chars/sec")
    
    elif generate_btn:
        st.warning(" Please enter some text to generate handwriting.")
    
    # Advanced Features Section
    st.markdown("---")
    st.markdown("##  Advanced Features")
    
    feature_tabs = st.tabs([" Analytics Dashboard", " Batch Comparison", " Model Insights", "Export Options"])
    
    with feature_tabs[0]:
        st.markdown("###  Generation Analytics")
        
        if st.session_state.analytics.generation_history:
            analytics_fig = st.session_state.analytics.get_analytics_dashboard()
            if analytics_fig:
                st.plotly_chart(analytics_fig, use_container_width=True)
            
            # Recent generations table
            st.markdown("#### Recent Generations")
            recent_df = pd.DataFrame(st.session_state.analytics.generation_history[-10:])
            if not recent_df.empty:
                display_df = recent_df[['timestamp', 'text', 'style', 'method', 'quality_score']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
                display_df['text'] = display_df['text'].str[:30] + '...'
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info(" Generate some handwriting to see analytics!")
    
    with feature_tabs[1]:
        st.markdown("###  Style Comparison Tool")
        
        comparison_text = st.text_input("Text for comparison:", "Compare styles")
        
        if st.button("Generate Style Comparison") and comparison_text:
            comparison_cols = st.columns(3)
            
            for i, comp_style in enumerate(['casual', 'formal', 'cursive']):
                with comparison_cols[i]:
                    with st.spinner(f"Generating {comp_style}..."):
                        comp_img = create_ultra_realistic_handwriting(comparison_text, comp_style)
                        st.image(comp_img, caption=f"{comp_style.title()} Style")
                        
                        quality_score, _ = quality_assessor.assess_image_quality(comp_img)
                        st.metric("Quality Score", f"{quality_score:.3f}")
    
    with feature_tabs[2]:
        st.markdown("###  Model Architecture Insights")
        
        st.markdown("""
        ####  Advanced Generator Architecture
        - **Multi-scale Generation**: Supports 28x28, 56x56, and 112x112 outputs
        - **Attention Mechanism**: Self-attention for better feature learning
        - **Style Conditioning**: 50-dimensional style vectors
        - **Linguistic Features**: 256-dimensional text encoding with linguistic analysis
        
        #### Style Analysis System
        - **7 Style Parameters**: Slant, thickness, spacing, irregularity, pressure, speed, fluidity
        - **Dynamic Style Vectors**: Real-time style parameter generation
        - **Style Transfer**: Cross-style feature blending capabilities
        
        ####  Quality Assessment Engine
        - **Multi-metric Evaluation**: Sharpness, contrast, edge density, noise analysis
        - **Real-time Quality Scoring**: Instant feedback on generation quality
        - **Performance Analytics**: Generation time and efficiency tracking
        """)
        
        # Model statistics
        model_stats = {
            'Generator Parameters': '~2.5M',
            'Text Encoder Features': '256',
            'Style Vector Dimensions': '50',
            'Supported Resolutions': '3 (28x28, 56x56, 112x112)',
            'Quality Metrics': '4 (Sharpness, Contrast, Edge, Noise)',
            'Style Variations': '6 + Custom'
        }
        
        stats_df = pd.DataFrame(list(model_stats.items()), columns=['Component', 'Specification'])
        st.table(stats_df)
    
    with feature_tabs[3]:
        st.markdown("###  Export & Integration Options")
        
        export_cols = st.columns(2)
        
        with export_cols[0]:
            st.markdown("####  API Integration")
            st.code("""
# Example API usage
import requests

response = requests.post(
    'http://your-api/generate',
    json={
        'text': 'Your text here',
        'style': 'casual',
        'quality': 'high',
        'seed': 42
    }
)
            """, language='python')
        
        with export_cols[1]:
            st.markdown("####  Batch Export")
            if st.button("Export All Analytics Data"):
                if st.session_state.analytics.generation_history:
                    export_df = pd.DataFrame(st.session_state.analytics.generation_history)
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        " Download Analytics CSV",
                        data=csv_data,
                        file_name=f"handwriting_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No analytics data to export yet.")
    
    # Footer with project information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
        <h4> Advanced AI Handwriting Synthesis Project</h4>
        <p><strong>Key Technologies:</strong> PyTorch, Streamlit, Computer Vision, NLP, GAN Architecture</p>
        <p><strong>Advanced Features:</strong> Linguistic Analysis, Style Transfer, Quality Assessment, Real-time Analytics</p>
        <p><strong>Innovation Highlights:</strong> Multi-scale Generation, Attention Mechanisms, Custom Style Vectors</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()