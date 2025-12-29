# Seal Detection - Docker Environment

シール検出用のDocker環境です。2つの検出方式を提供しています。

## 前提条件

- **YOLOX**: 既にクローン済みであること
- **CUDA**: 12.6
- **Docker**: Docker Compose v2対応
- **NVIDIA GPU**: 必須

## 対象シール

| クラス | 色 | 形状 |
|--------|-----|------|
| blue_triangle | 青 | 三角形 |
| yellow_octagon | 黄色 | 八角形 |

## 検出方式

### 方式1: HSV色検出 + 輪郭形状分析（学習不要・推奨）

色と形状が明確なシールに最適。すぐに使用可能。

```
入力 → HSV変換 → 色マスク → 輪郭検出 → 頂点数で分類
```

### 方式2: YOLOX（深層学習）

環境変化が激しい場合や、より高精度が必要な場合に使用。
学習データの準備が必要。

## ディレクトリ構成

```
yolox-docker/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── YOLOX/                      # 既存のYOLOXクローン（マウント）
├── scripts/
│   ├── # HSV色検出（学習不要）
│   ├── color_shape_detector.py # 検出クラス
│   ├── detect_video.py         # 動画/画像検出
│   ├── calibrate_hsv.py        # HSVパラメータ調整
│   ├── detector_config.yaml    # 検出設定
│   │
│   ├── # YOLOX（深層学習）
│   ├── yolox_train.sh          # 学習スクリプト
│   ├── yolox_inference.py      # 推論スクリプト
│   └── yolox_prepare_dataset.py # データセット準備
├── exps/                       # YOLOX実験設定
│   └── seal_detection_exp.py
├── dataset/                    # YOLOX学習データ
├── weights/                    # YOLOX事前学習済み重み
├── outputs/                    # YOLOX学習出力
├── videos/                     # 入力/出力動画
└── notebooks/                  # Jupyterノートブック
```

## クイックスタート

### 1. 環境構築

```bash
cd yolox-docker

# YOLOXをクローン（まだの場合）
git clone https://github.com/Megvii-BaseDetection/YOLOX.git

# 初期セットアップ
bash setup.sh

# ビルド・起動
docker-compose build
docker-compose up -d
docker-compose exec yolox bash
```

---

## 方式1: HSV色検出（学習不要）

### 基本的な使い方

```bash
cd /workspace/scripts

# 動画から検出
python detect_video.py \
    -i /workspace/videos/input.mp4 \
    -o /workspace/videos/output.mp4 \
    --show

# 画像から検出
python detect_video.py \
    -i /workspace/videos/frame.jpg \
    --vertices \
    --show
```

### HSVパラメータ調整

実際のシールの色に合わせてパラメータを調整：

```bash
python calibrate_hsv.py -i /workspace/videos/sample.jpg
```

**操作方法：**
| キー | 動作 |
|------|------|
| q | 終了 |
| p | 現在の値を表示 |
| s | 設定をファイルに保存 |
| b | 青のプリセット |
| y | 黄のプリセット |
| r | リセット |

### 設定ファイル

`detector_config.yaml` を編集：

```yaml
blue:
  hsv_lower: [100, 100, 100]
  hsv_upper: [130, 255, 255]
  expected_vertices: 3
  vertex_tolerance: 1

yellow:
  hsv_lower: [20, 100, 100]
  hsv_upper: [35, 255, 255]
  expected_vertices: 8
  vertex_tolerance: 2

detection:
  min_area: 500
  max_area: 100000
```

---

## 方式2: YOLOX（深層学習）

### データセット準備

```bash
cd /workspace/scripts

# 動画からフレーム抽出
python yolox_prepare_dataset.py extract \
    -v /workspace/videos/input.mp4 \
    -o /workspace/dataset/raw_images \
    -i 30

# アノテーション後、データセット分割
python yolox_prepare_dataset.py split \
    -i /workspace/dataset/raw_images \
    -a /workspace/dataset/annotations.json \
    -o /workspace/dataset \
    -r 0.8

# 検証
python yolox_prepare_dataset.py verify -d /workspace/dataset
```

### 学習

```bash
cd /workspace/YOLOX

# 事前学習済み重みダウンロード
wget -P /workspace/weights \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

# 学習実行
bash /workspace/scripts/yolox_train.sh

# オプション指定
bash /workspace/scripts/yolox_train.sh -b 32 --fp16
```

### 推論

```bash
python /workspace/scripts/yolox_inference.py \
    -c /workspace/outputs/seal_detection/best_ckpt.pth \
    -v /workspace/videos/test.mp4 \
    -o /workspace/videos/output.mp4 \
    --show
```

---

## Jupyter Notebook

```bash
# コンテナ内で起動
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ブラウザでアクセス: http://localhost:8888
```

## よく使うコマンド

```bash
docker-compose exec yolox bash    # コンテナに入る
docker-compose logs -f            # ログ確認
docker-compose down               # 停止
docker-compose build --no-cache   # 再ビルド
```

## どちらを使うべきか？

| 条件 | 推奨方式 |
|------|----------|
| すぐに使いたい | HSV色検出 |
| 色と形状が明確 | HSV色検出 |
| 照明条件が安定 | HSV色検出 |
| 環境変化が激しい | YOLOX |
| より高精度が必要 | YOLOX |
| 学習データを用意できる | YOLOX |

## トラブルシューティング

### HSV検出で検出されない
1. `calibrate_hsv.py` でHSV範囲を確認
2. `min_area` を小さくする
3. S, V範囲を広げる

### YOLOX学習が進まない
- データセットのアノテーション数を確認（各クラス50枚以上推奨）
- バッチサイズを下げる

## ライセンス

YOLOX: Apache License 2.0
