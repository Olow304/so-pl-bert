#!/bin/bash
# Download missing pre-trained models for StyleTTS2

echo "Downloading pre-trained ASR and JDC models..."

cd StyleTTS2

# Download ASR model
if [ ! -f "Utils/ASR/epoch_00080.pth" ]; then
    echo "Downloading ASR model..."
    mkdir -p Utils/ASR
    wget -O Utils/ASR/epoch_00080.pth https://github.com/yl4579/AuxiliaryASR/releases/download/v0.0.1/epoch_00080.pth
fi

# Download JDC model
if [ ! -f "Utils/JDC/bst.t7" ]; then
    echo "Downloading JDC model..."
    mkdir -p Utils/JDC
    wget -O Utils/JDC/bst.t7 https://github.com/yl4579/PitchExtractor/releases/download/v0.0.1/bst.t7
fi

cd ..

echo "Downloads complete!"