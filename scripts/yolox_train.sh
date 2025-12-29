#!/bin/bash
# YOLOX Training Script for Seal Detection

# デフォルト設定
EXP_FILE="/workspace/exps/custom/seal_detection_exp.py"
BATCH_SIZE=16
DEVICES=0
PRETRAINED_WEIGHTS="/workspace/weights/yolox_s.pth"

# 使い方表示
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --exp FILE        Experiment file (default: $EXP_FILE)"
    echo "  -b, --batch SIZE      Batch size (default: $BATCH_SIZE)"
    echo "  -d, --devices IDS     GPU device IDs (default: $DEVICES)"
    echo "  -w, --weights FILE    Pretrained weights (default: $PRETRAINED_WEIGHTS)"
    echo "  --resume CKPT         Resume from checkpoint"
    echo "  --fp16                Enable FP16 training"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  # 基本的な学習"
    echo "  $0"
    echo ""
    echo "  # バッチサイズとGPU指定"
    echo "  $0 -b 32 -d 0,1"
    echo ""
    echo "  # 学習再開"
    echo "  $0 --resume /workspace/outputs/seal_detection/latest_ckpt.pth"
}

# 引数解析
FP16=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--exp)
            EXP_FILE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--devices)
            DEVICES="$2"
            shift 2
            ;;
        -w|--weights)
            PRETRAINED_WEIGHTS="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume --ckpt $2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# GPU数をカウント
NUM_GPUS=$(echo $DEVICES | tr ',' '\n' | wc -l)

echo "=============================================="
echo "YOLOX Training for Seal Detection"
echo "=============================================="
echo "Experiment file: $EXP_FILE"
echo "Batch size: $BATCH_SIZE"
echo "GPU devices: $DEVICES (count: $NUM_GPUS)"
echo "Pretrained weights: $PRETRAINED_WEIGHTS"
echo "FP16: ${FP16:-disabled}"
echo "Resume: ${RESUME:-disabled}"
echo "=============================================="

# 事前学習済み重みのダウンロード（存在しない場合）
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "Downloading pretrained weights..."
    mkdir -p /workspace/weights
    wget -O "$PRETRAINED_WEIGHTS" \
        https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
fi

# 学習実行
export CUDA_VISIBLE_DEVICES=$DEVICES

if [ $NUM_GPUS -gt 1 ]; then
    # マルチGPU学習
    python -m yolox.tools.train \
        -f "$EXP_FILE" \
        -d $NUM_GPUS \
        -b $BATCH_SIZE \
        -c "$PRETRAINED_WEIGHTS" \
        --occupy \
        $FP16 \
        $RESUME
else
    # シングルGPU学習
    python tools/train.py \
        -f "$EXP_FILE" \
        -d $NUM_GPUS \
        -b $BATCH_SIZE \
        -c "$PRETRAINED_WEIGHTS" \
        --occupy \
        $FP16 \
        $RESUME
fi

echo "Training completed!"
