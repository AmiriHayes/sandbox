import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# 1. DATA DOWNLOADING AND LOADING
# ================================

def download_flickr8k_dataset():
    """Download Flickr8k dataset using Kaggle API"""
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    # Configure Kaggle credentials
    os.environ["KAGGLE_USERNAME"] = "hayz07"
    os.environ["KAGGLE_KEY"] = "d71fe83e4b69f92ad3f1f0b55be98a96"
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    dataset_dir = "./flickr8k"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        api.dataset_download_files("adityajn105/flickr8k", path=dataset_dir, unzip=True)
    
    print("Dataset downloaded to:", dataset_dir)
    return dataset_dir

def load_captions(captions_file):
    """Load and parse captions from file"""
    captions_dict = {}
    
    with open(captions_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                img_name, caption = parts
                # Remove .jpg#0, .jpg#1, etc. to get base image name
                img_base = img_name.split('#')[0]
                
                if img_base not in captions_dict:
                    captions_dict[img_base] = []
                captions_dict[img_base].append(caption.strip('"'))
    
    return captions_dict

# ================================
# 2. DATA PREPROCESSING
# ================================

class Vocabulary:
    """Vocabulary class for captions"""
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        """Simple English tokenizer"""
        return [tok.lower().strip() for tok in re.findall(r'\b\w+\b', text)]

    def build_vocabulary(self, sentence_list):
        """Build vocabulary from list of sentences"""
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """Convert text to numerical sequence"""
        tokenized_text = self.tokenizer_eng(text)
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def preprocess_captions(captions_dict):
    """Preprocess captions by cleaning and normalizing"""
    processed_captions = {}
    
    for img_name, captions in captions_dict.items():
        processed_list = []
        for caption in captions:
            # Convert to lowercase and remove extra whitespace
            caption = caption.lower().strip()
            # Remove punctuation except periods
            caption = re.sub(r'[^\w\s]', '', caption)
            # Remove extra whitespace
            caption = re.sub(r'\s+', ' ', caption)
            processed_list.append(caption)
        processed_captions[img_name] = processed_list
    
    return processed_captions

# ================================
# 3. DATASET CLASS
# ================================

class FlickrDataset(Dataset):
    """Custom dataset for Flickr8k"""
    def __init__(self, root_dir, captions_dict, vocab, transform=None, max_length=50):
        self.root_dir = root_dir
        self.captions_dict = captions_dict
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
        # Create list of (image_path, caption) pairs
        self.data_pairs = []
        for img_name, captions in captions_dict.items():
            img_path = os.path.join(root_dir, img_name)
            if os.path.exists(img_path):
                for caption in captions:
                    self.data_pairs.append((img_path, caption))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, caption = self.data_pairs[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert caption to numerical sequence
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        # Truncate if too long
        if len(numericalized_caption) > self.max_length:
            numericalized_caption = numericalized_caption[:self.max_length]
        
        return image, torch.tensor(numericalized_caption)

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

# ================================
# 4. MODEL ARCHITECTURE
# ================================

class EncoderCNN(nn.Module):
    """CNN Encoder using ResNet50"""
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(pretrained=True)
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.dropout(features)
        return features

class DecoderRNN(nn.Module):
    """RNN Decoder with attention mechanism"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, seq_length)
        
        batch_size = features.size(0)
        
        # Initialize hidden state with image features
        h0 = features.unsqueeze(1).repeat(1, self.num_layers, 1).transpose(0, 1).contiguous()
        c0 = torch.zeros_like(h0)
        
        # Embed captions (exclude last token for input)
        embeddings = self.embedding(captions[:, :-1])
        
        # Concatenate image features with embedded captions
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(self.dropout(lstm_out))
        
        return outputs

class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=50):
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            
            # Initialize with SOS token
            inputs = torch.tensor([vocab.stoi["<SOS>"]]).unsqueeze(0).to(device)
            hidden = features.unsqueeze(1).repeat(1, self.decoder.num_layers, 1).transpose(0, 1).contiguous()
            cell = torch.zeros_like(hidden)
            
            result = []
            
            for _ in range(max_length):
                embeddings = self.decoder.embedding(inputs)
                lstm_out, (hidden, cell) = self.decoder.lstm(embeddings, (hidden, cell))
                outputs = self.decoder.linear(lstm_out)
                predicted = outputs.argmax(2)
                
                predicted_word = vocab.itos[predicted.item()]
                
                if predicted_word == "<EOS>":
                    break
                    
                result.append(predicted_word)
                inputs = predicted
                
            return " ".join(result)

# ================================
# 5. TRAINING FUNCTIONS
# ================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, vocab):
    """Train the image captioning model"""
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch_idx, (images, captions) in enumerate(train_pbar):
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, captions)
            
            # Calculate loss (exclude SOS token from targets)
            targets = captions[:, 1:].contiguous().view(-1)
            outputs = outputs.view(-1, outputs.size(2))
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
            for images, captions in val_pbar:
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions)
                targets = captions[:, 1:].contiguous().view(-1)
                outputs = outputs.view(-1, outputs.size(2))
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print('-' * 50)
        
        # Generate sample caption
        if (epoch + 1) % 5 == 0:
            sample_image = next(iter(val_loader))[0][0].to(device)
            sample_caption = model.generate_caption(sample_image, vocab)
            print(f'Sample Caption: {sample_caption}')
            print('-' * 50)
    
    return train_losses, val_losses

# ================================
# 6. EVALUATION FUNCTIONS
# ================================

def calculate_bleu_score(model, data_loader, vocab):
    """Calculate BLEU score on dataset"""
    model.eval()
    references = []
    candidates = []
    
    with torch.no_grad():
        for images, captions in tqdm(data_loader, desc="Calculating BLEU"):
            for i in range(images.size(0)):
                image = images[i].to(device)
                
                # Generate caption
                generated_caption = model.generate_caption(image, vocab)
                generated_tokens = generated_caption.split()
                
                # Get ground truth caption
                caption_tokens = []
                caption_indices = captions[i].tolist()
                for idx in caption_indices:
                    if idx == vocab.stoi["<EOS>"]:
                        break
                    if idx not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"]]:
                        caption_tokens.append(vocab.itos[idx])
                
                candidates.append(generated_tokens)
                references.append([caption_tokens])
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu1, bleu2, bleu3, bleu4

def visualize_predictions(model, data_loader, vocab, num_samples=5):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        images, captions = next(iter(data_loader))
        
        for i in range(min(num_samples, images.size(0))):
            image = images[i].to(device)
            
            # Generate caption
            generated_caption = model.generate_caption(image, vocab)
            
            # Get ground truth caption
            caption_tokens = []
            caption_indices = captions[i].tolist()
            for idx in caption_indices:
                if idx == vocab.stoi["<EOS>"]:
                    break
                if idx not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"]]:
                    caption_tokens.append(vocab.itos[idx])
            ground_truth = " ".join(caption_tokens)
            
            # Display image and captions
            img_np = images[i].permute(1, 2, 0).numpy()
            # Denormalize if necessary
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            axes[i].imshow(img_np)
            axes[i].set_title(f'Generated: {generated_caption}\nGround Truth: {ground_truth}', 
                             wrap=True, fontsize=10)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_training_progress(train_losses, val_losses):
    """Visualize training progress"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_dataset(captions_dict):
    """Analyze and visualize dataset statistics"""
    # Caption length analysis
    lengths = []
    all_words = []
    
    for captions in captions_dict.values():
        for caption in captions:
            words = caption.split()
            lengths.append(len(words))
            all_words.extend(words)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Caption length distribution
    axes[0, 0].hist(lengths, bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Caption Length Distribution')
    axes[0, 0].set_xlabel('Number of Words')
    axes[0, 0].set_ylabel('Frequency')
    
    # Word frequency
    word_counts = Counter(all_words)
    top_words = dict(word_counts.most_common(20))
    axes[0, 1].bar(range(len(top_words)), list(top_words.values()))
    axes[0, 1].set_title('Top 20 Most Frequent Words')
    axes[0, 1].set_xlabel('Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xticks(range(len(top_words)))
    axes[0, 1].set_xticklabels(list(top_words.keys()), rotation=45)
    
    # Dataset statistics
    stats_text = f"""
    Dataset Statistics:
    - Total images: {len(captions_dict)}
    - Total captions: {sum(len(caps) for caps in captions_dict.values())}
    - Average caption length: {np.mean(lengths):.2f} words
    - Max caption length: {max(lengths)} words
    - Min caption length: {min(lengths)} words
    - Unique words: {len(set(all_words))}
    """
    
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Sample images grid
    axes[1, 1].text(0.5, 0.5, 'Dataset loaded successfully!\nReady for training.', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ================================
# 7. MAIN PIPELINE
# ================================

def main():
    """Main pipeline execution"""
    print("Starting Image Captioning Pipeline...")
    
    # 1. Download dataset
    print("\n1. Downloading dataset...")
    dataset_dir = download_flickr8k_dataset()
    
    # 2. Load captions
    print("\n2. Loading captions...")
    captions_file = os.path.join(dataset_dir, "captions.txt")
    captions_dict = load_captions(captions_file)
    print(f"Loaded {len(captions_dict)} images with captions")
    
    # 3. Preprocess captions
    print("\n3. Preprocessing captions...")
    captions_dict = preprocess_captions(captions_dict)
    
    # 4. Analyze dataset
    print("\n4. Analyzing dataset...")
    analyze_dataset(captions_dict)
    
    # 5. Build vocabulary
    print("\n5. Building vocabulary...")
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 6. Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # 7. Data splitting
    print("\n6. Splitting data...")
    image_names = list(captions_dict.keys())
    train_imgs, temp_imgs = train_test_split(image_names, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    train_captions = {img: captions_dict[img] for img in train_imgs}
    val_captions = {img: captions_dict[img] for img in val_imgs}
    test_captions = {img: captions_dict[img] for img in test_imgs}
    
    print(f"Train: {len(train_captions)}, Val: {len(val_captions)}, Test: {len(test_captions)}")
    
    # 8. Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 9. Create datasets
    print("\n7. Creating datasets...")
    images_dir = os.path.join(dataset_dir, "Images")
    
    train_dataset = FlickrDataset(images_dir, train_captions, vocab, transform)
    val_dataset = FlickrDataset(images_dir, val_captions, vocab, transform)
    test_dataset = FlickrDataset(images_dir, test_captions, vocab, transform)
    
    # 10. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4)
    
    # 11. Initialize model
    print("\n8. Initializing model...")
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    num_layers = 1
    
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
    model = model.to(device)
    
    # 12. Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # 13. Training
    print("\n9. Training model...")
    num_epochs = 20
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         criterion, optimizer, num_epochs, vocab)
    
    # 14. Save model
    torch.save(model.state_dict(), 'image_captioning_model.pth')
    print("Model saved!")
    
    # 15. Visualization
    print("\n10. Visualizing results...")
    visualize_training_progress(train_losses, val_losses)
    visualize_predictions(model, test_loader, vocab)
    
    # 16. Evaluation
    print("\n11. Evaluating model...")
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu_score(model, test_loader, vocab)
    
    print(f"\nBLEU Scores:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()