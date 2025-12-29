# YOLOX Docker Environment for Seal Detection
# GPU対応 (CUDA 12.6 + PyTorch 2.5)
# 前提: YOLOXリポジトリは既にクローン済みでマウントして使用

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /workspace

# Python依存関係のインストール
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# YOLOXはマウントされる前提なので、起動時にインストール
# コンテナ起動後に /workspace/YOLOX で pip install -e . を実行

# カスタムコードとデータ用ディレクトリ
RUN mkdir -p /workspace/dataset \
    /workspace/exps/custom \
    /workspace/outputs \
    /workspace/scripts \
    /workspace/weights

# 作業ディレクトリ
WORKDIR /workspace

# デフォルトコマンド
CMD ["/bin/bash"]
