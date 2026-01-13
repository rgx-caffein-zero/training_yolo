# Seal Detection - Docker Environment

シール検出用のDocker環境です。2つの検出方式を提供しています。

## 前提条件

- **YOLOX**: 既にクローン済みであること
- **CUDA**: 12.6
- **Docker**: Docker Compose v2対応
- **NVIDIA GPU**: 必須

## 対象シール

| クラス | 色 | 形状 | 用途 |
|--------|-----|------|------|
| blue_triangle | 青 | 三角形 | 冷蔵（2〜10℃） |
| yellow_octagon | 黄色 | 八角形 | 冷凍（-18℃以下） |

## 検出方式

### 方式1: HSV色検出 + 輪郭形状分析（学習不要・推奨）

色と形状が明確なシールに最適。すぐに使用可能。

```
入力 → HSV変換 → 色マスク → 輪郭検出 → 凸包処理 → 頂点数で分類
```

**特徴:**
- 学習不要ですぐに使える
- 凸包（Convex Hull）処理で内部の白い領域を無視
- 近接領域の自動マージ機能

### 方式2: YOLOX（深層学習）

環境変化が激しい場合や、より高精度が必要な場合に使用。
学習データの準備が必要。

**特徴:**
- COCO事前学習モデル（80クラス）での汎用検出にも対応
- カスタム学習でシール専用モデルを作成可能

## ディレクトリ構成

```
seal-detection/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── cvat/                       # 既存のcvatクローン（マウント）
├── YOLOX/                      # 既存のYOLOXクローン（マウント）
├── scripts/
│   ├── # HSV色検出（学習不要）
│   ├── color_shape_detector.py # 検出クラス
│   ├── detect_video.py         # 動画/画像検出（CLI）
│   ├── calibrate_hsv.py        # HSVパラメータ調整（GUI必要）
│   ├── detector_config.yaml    # 検出設定
│   │
│   ├── # YOLOX（深層学習）
│   ├── yolox_train.sh          # 学習スクリプト
│   ├── yolox_inference.py      # 推論スクリプト（COCO対応）
│   ├── yolox_prepare_dataset.py # データセット準備
│   ├── coco_classes.txt        # COCOクラス名（80クラス）
│   └── seal_classes.txt        # シールクラス名（2クラス）
├── notebooks/                  # Jupyter Notebook
│   ├── hsv_calibration.ipynb   # HSVキャリブレーション（GUI不要）
│   └── detection_test.ipynb    # 検出テスト・可視化
├── exps/                       # YOLOX実験設定
│   └── seal_detection_exp.py
├── dataset/                    # YOLOX学習データ
├── weights/                    # YOLOX事前学習済み重み
├── outputs/                    # YOLOX学習出力
└── videos/                     # 入力/出力動画
```

## クイックスタート

### 1. 環境構築

```bash
# YOLOXをクローン（まだの場合）
git clone https://github.com/Megvii-BaseDetection/YOLOX.git

# 初期セットアップ
bash setup.sh

# ビルド・起動
docker-compose build
docker-compose up -d

# ログを確認（YOLOXインストール完了を待つ）
docker logs -f yolox

# コンテナに入る
docker-compose exec yolox bash
```

### 2. 事前学習済み重みのダウンロード（YOLOX使用時）

```bash
# コンテナ内で実行
wget -P /workspace/weights \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

---

## 方式1: HSV色検出（学習不要）

### CLI版（GUI環境が必要）

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
    -o /workspace/videos/result.jpg \
    --vertices
```

### Jupyter Notebook版（GUI不要・推奨）

WSL2などGUI環境がない場合はJupyter Notebookを使用します。

```bash
# コンテナ内でJupyterを起動
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ブラウザでアクセス: http://localhost:8888
```

#### 提供ノートブック

| ノートブック | 用途 |
|-------------|------|
| `hsv_calibration.ipynb` | HSVパラメータの調整（インタラクティブスライダー） |
| `detection_test.ipynb` | 検出テスト・動画処理・バッチ処理 |

**hsv_calibration.ipynb の機能:**
- インタラクティブなスライダーでH/S/V値を調整
- リアルタイムでマスクと検出結果をプレビュー
- プリセットボタン（Blue/Yellow/Reset）
- 設定をYAMLファイルに保存

**detection_test.ipynb の機能:**
- 画像/動画からの検出テスト
- 検出結果の可視化（マスク表示付き）
- 動画フレームのインタラクティブ探索（スライダー）
- バッチ処理・出力生成

### 設定ファイル

`detector_config.yaml` を編集：

```yaml
# 青い三角形シール（冷蔵）
blue:
  hsv_lower: [95, 100, 100]
  hsv_upper: [130, 255, 255]
  expected_vertices: 4      # 凸包処理後の頂点数
  vertex_tolerance: 3
  class_name: blue_triangle

# 黄色い八角形シール（冷凍）
yellow:
  hsv_lower: [20, 100, 100]
  hsv_upper: [45, 255, 255]
  expected_vertices: 6      # 凸包処理後の頂点数
  vertex_tolerance: 4
  class_name: yellow_octagon

# 検出パラメータ
detection:
  min_area: 5000            # 最小面積
  max_area: 500000          # 最大面積
  epsilon_ratio: 0.03       # 輪郭近似精度
  morph_kernel_size: 7      # モルフォロジーカーネル
  blur_kernel_size: 5       # ブラーカーネル
  
  # 白い領域対策（シール内の品名欄・バーコード対応）
  use_convex_hull: true     # 凸包を使用
  fill_holes: true          # 内部の穴を埋める
  morph_iterations: 2       # モルフォロジー反復回数
  dilate_iterations: 5      # 膨張処理反復回数
```

---

## 方式2: YOLOX（深層学習）

### 推論（事前学習済みモデル）

学習なしで、COCO事前学習済みモデル（80クラス）を使って推論できます。

```bash
cd /workspace/scripts

# COCO 80クラスで検出
python yolox_inference.py \
    -c /workspace/weights/yolox_s.pth \
    -m yolox_s \
    -i /workspace/videos/input.mp4 \
    -o /workspace/videos/output.mp4

# 画像から検出
python yolox_inference.py \
    -c /workspace/weights/yolox_s.pth \
    -m yolox_s \
    -i /workspace/videos/sample.jpg \
    -o /workspace/videos/result.jpg

# 特定クラスのみ検出（フィルタ機能）
python yolox_inference.py \
    -c /workspace/weights/yolox_s.pth \
    -m yolox_s \
    -i /workspace/videos/input.mp4 \
    -o /workspace/videos/output.mp4 \
    --filter person,car,truck

# COCOクラス一覧を表示
python yolox_inference.py --list-classes
```

#### 推論オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `-c, --ckpt` | チェックポイントファイル | 必須 |
| `-m, --model` | YOLOXモデル名 | - |
| `-e, --exp` | 実験設定ファイル（カスタムモデル用） | - |
| `-i, --input` | 入力ファイル（動画/画像） | 必須 |
| `-o, --output` | 出力ファイル | - |
| `--conf` | 信頼度閾値 | 0.5 |
| `--nms` | NMS閾値 | 0.45 |
| `--size` | 入力サイズ | 640 |
| `--filter` | 検出クラスをフィルタ（カンマ区切り） | - |
| `--num-classes` | クラス数 | 80 |
| `--class-names` | クラス名ファイル | - |
| `--list-classes` | COCOクラス一覧を表示 | - |

#### 対応モデル

| モデル名 | パラメータ数 | 特徴 |
|---------|------------|------|
| `yolox_nano` | 0.91M | 最速・軽量 |
| `yolox_tiny` | 5.06M | 高速 |
| `yolox_s` | 9.0M | バランス（推奨） |
| `yolox_m` | 25.3M | 高精度 |
| `yolox_l` | 54.2M | 高精度 |
| `yolox_x` | 99.1M | 最高精度 |

### カスタム学習（シール検出専用モデル）

#### データセット準備

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

#### 学習

```bash
cd /workspace/YOLOX

# 学習実行
bash /workspace/scripts/yolox_train.sh

# オプション指定
bash /workspace/scripts/yolox_train.sh -b 32 --fp16
```

#### 推論（カスタムモデル）

```bash
python /workspace/scripts/yolox_inference.py \
    -c /workspace/outputs/seal_detection/best_ckpt.pth \
    -e /workspace/exps/custom/seal_detection_exp.py \
    -i /workspace/videos/test.mp4 \
    -o /workspace/videos/output.mp4
```

---

## Jupyter Notebook

```bash
# コンテナ内で起動
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ブラウザでアクセス: http://localhost:8888
```

| ノートブック | 説明 |
|-------------|------|
| `hsv_calibration.ipynb` | HSVパラメータ調整（GUI不要） |
| `detection_test.ipynb` | 検出テスト・動画処理 |

---

## よく使うコマンド

```bash
# コンテナに入る
docker-compose exec yolox bash

# ログ確認
docker logs -f yolox

# 停止
docker-compose down

# 再ビルド（キャッシュなし）
docker-compose build --no-cache

# YOLOXの再インストール（問題発生時）
docker-compose down -v
rm -f .yolox_installed
docker-compose up -d
```

---

## どちらを使うべきか？

| 条件 | 推奨方式 |
|------|----------|
| すぐに使いたい | HSV色検出 |
| 色と形状が明確 | HSV色検出 |
| 照明条件が安定 | HSV色検出 |
| 環境変化が激しい | YOLOX |
| より高精度が必要 | YOLOX |
| 汎用物体検出（人、車など） | YOLOX（COCO） |

---

## トラブルシューティング

### Docker関連

#### `onnx-simplifier` インストールエラー
```
packaging.version.InvalidVersion: Invalid version: 'unknown'
```

**解決:** `docker-compose.yml` が最新版か確認。YOLOXの `setup.py` を自動修正する処理が含まれています。

```bash
# 完全にリセットして再起動
docker-compose down -v
docker-compose up -d
docker logs -f yolox
```

### HSV検出関連

#### 検出されない
1. `hsv_calibration.ipynb` でHSV範囲を確認
2. `min_area` を小さくする（例: 5000 → 1000）
3. S, V範囲を広げる（min値を下げる）

#### シールが複数に分割されて検出される
1. `dilate_iterations` を増やす（例: 5 → 8）
2. `morph_iterations` を増やす（例: 2 → 3）

#### 誤検出が多い
1. `min_area` を大きくする
2. HSV範囲を狭める
3. `vertex_tolerance` を小さくする

### YOLOX関連

#### 学習が進まない
- データセットのアノテーション数を確認（各クラス50枚以上推奨）
- バッチサイズを下げる（`-b 8`）

#### 推論でクラス名が `class_0` などになる
- `--class-names` でクラス名ファイルを指定
- `--num-classes` でクラス数を正しく指定

---

## ライセンス

- YOLOX: Apache License 2.0
- このプロジェクト: MIT License
