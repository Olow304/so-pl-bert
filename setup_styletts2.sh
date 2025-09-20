#!/bin/bash
# Setup script for StyleTTS2 with Somali PL-BERT

echo "=========================================="
echo "Setting up StyleTTS2 for Somali TTS"
echo "=========================================="

# 1. Clone StyleTTS2 if not present
if [ ! -d "StyleTTS2" ]; then
    echo "→ Cloning StyleTTS2 repository..."
    git clone https://github.com/yl4579/StyleTTS2.git
    cd StyleTTS2
    pip install -r requirements.txt
    cd ..
else
    echo "✓ StyleTTS2 already exists"
fi

# 2. Create Models directory structure
echo "→ Creating model directories..."
mkdir -p Models/LibriTTS

# 3. Download pre-trained LibriTTS model from HuggingFace
cd Models/LibriTTS

if [ ! -f "epochs_2nd_00020.pth" ]; then
    echo "→ Downloading pre-trained LibriTTS model from HuggingFace..."

    # Download the checkpoint (771 MB) from the Models/LibriTTS folder
    wget -c https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth

    # Download the config from the Models/LibriTTS folder
    wget -c https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml

    echo "✓ Downloaded pre-trained model"
else
    echo "✓ Pre-trained model already exists"
fi

cd ../..

# 4. Install additional dependencies
echo "→ Installing additional dependencies..."
pip install phonemizer librosa einops einops-exts torchaudio

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare data:    python prepare_styletts2_data.py"
echo "2. Start training:  python finetune_styletts2_somali.py"
echo ""