#!/bin/bash
# Initial setup script for Seal Detection

echo "=============================================="
echo "Seal Detection - Setup"
echo "=============================================="

# YOLOXの存在確認
if [ ! -d "YOLOX" ]; then
    echo ""
    echo "WARNING: YOLOX directory not found!"
    echo "Please clone YOLOX first:"
    echo "  git clone https://github.com/Megvii-BaseDetection/YOLOX.git"
    echo ""
    read -p "Clone YOLOX now? (y/n): " answer
    if [ "$answer" = "y" ]; then
        git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    fi
fi

echo "Setting up directory structure..."

# ディレクトリ作成
mkdir -p videos
mkdir -p notebooks
# YOLOX用
mkdir -p dataset/train
mkdir -p dataset/val
mkdir -p dataset/annotations
mkdir -p weights
mkdir -p outputs

# 実行権限付与
chmod +x scripts/*.sh 2>/dev/null

# .gitkeepファイル作成
touch videos/.gitkeep
touch notebooks/.gitkeep
touch dataset/train/.gitkeep
touch dataset/val/.gitkeep
touch dataset/annotations/.gitkeep
touch weights/.gitkeep
touch outputs/.gitkeep

echo ""
echo "Directory structure created!"
echo ""
echo "=============================================="
echo "Next steps:"
echo "=============================================="
echo "1. Build Docker image:"
echo "   docker-compose build"
echo ""
echo "2. Start container:"
echo "   docker-compose up -d"
echo ""
echo "3. Enter container:"
echo "   docker-compose exec yolox bash"
echo ""
echo "4. Run HSV detection (no training needed):"
echo "   cd /workspace/scripts"
echo "   python detect_video.py -i /workspace/videos/input.mp4 --show"
echo ""
echo "5. Or calibrate HSV parameters:"
echo "   python calibrate_hsv.py -i /workspace/videos/sample.jpg"
echo ""
echo "See README.md for YOLOX training instructions."
