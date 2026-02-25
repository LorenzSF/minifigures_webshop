#!/bin/bash

echo "============================================"
echo "  📦 Minifigures Dataset Setup"
echo "============================================"
echo ""

# Check if data already exists
if [ -d "./data/data/minifigures" ] && [ "$(ls -A ./data/data/minifigures 2>/dev/null)" ]; then
    echo "⚠️  Dataset already exists locally."
    read -p "Do you want to re-download it? (y/N): " REDOWNLOAD
    if [[ ! "$REDOWNLOAD" =~ ^[Yy]$ ]]; then
        echo "✅ Skipping download. Using existing dataset."
        exit 0
    fi
fi

echo ""
read -p "📥 Do you want to download the full dataset? (y/N): " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "⏭️  Skipping dataset download."
    echo "   Run './setup_data.sh' manually when you're ready."
    exit 0
fi

echo ""
echo "📁 Creating data directories..."
mkdir -p ./data/data

echo ""
echo "⬇️  Downloading minifigures dataset (this may take a while)..."
dvc get-url https://roai-data-readonly.s3.eu-central-1.amazonaws.com/minifigures.tar.gz ./data/data/minifigures.tar.gz

if [ $? -ne 0 ]; then
    echo "❌ Download failed. Please check your internet connection."
    exit 1
fi

echo ""
echo "📂 Extracting dataset..."
tar -xzf ./data/data/minifigures.tar.gz -C ./data/data

echo ""
echo "⬇️  Downloading dataset labels (dataset.json)..."
dvc get-url https://roai-data-readonly.s3.eu-central-1.amazonaws.com/dataset.json ./data/data/dataset.json

if [ $? -ne 0 ]; then
    echo "❌ Failed to download dataset.json. Please try again."
    exit 1
fi

echo ""
echo "🧹 Cleaning up compressed file..."
rm -f ./data/data/minifigures.tar.gz

echo ""
echo "============================================"
echo "  ✅ Dataset ready!"
echo "  📁 Location: ./data/data/minifigures"
echo "  📄 Labels:   ./data/data/dataset.json"
echo "  📊 Files:    $(ls ./data/data/minifigures/*.png 2>/dev/null | wc -l) images"
echo "============================================"
