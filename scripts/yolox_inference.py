#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX Inference Script (Extended Version)

YOLOXモデルを使用して動画/画像から物体を検出するスクリプト

対応モード:
1. シール検出モード（カスタム学習済みモデル）
2. COCO検出モード（事前学習済みモデル）
3. カスタムクラスモード（任意のクラス定義）
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


# COCO 80クラス
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# シール検出用クラス
SEAL_CLASSES = ['blue_triangle', 'yellow_octagon']

# YOLOXモデル設定
YOLOX_MODELS = {
    'yolox_nano': {'depth': 0.33, 'width': 0.25},
    'yolox_tiny': {'depth': 0.33, 'width': 0.375},
    'yolox_s': {'depth': 0.33, 'width': 0.50},
    'yolox_m': {'depth': 0.67, 'width': 0.75},
    'yolox_l': {'depth': 1.0, 'width': 1.0},
    'yolox_x': {'depth': 1.33, 'width': 1.25},
}


def generate_colors(num_classes):
    """クラスごとの色を生成"""
    np.random.seed(42)
    colors = {}
    for i in range(num_classes):
        colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))
    return colors


class YOLOXDetector:
    """汎用YOLOXベースの検出クラス"""
    
    def __init__(
        self,
        ckpt_path: str,
        exp_path: str = None,
        model_name: str = None,
        num_classes: int = None,
        class_names: list = None,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
        input_size: tuple = (640, 640),
        device: str = "auto"
    ):
        """
        Args:
            ckpt_path: チェックポイントファイルのパス
            exp_path: 実験設定ファイルのパス（カスタムモデル用）
            model_name: YOLOXモデル名（yolox_s, yolox_m等）- 事前学習済みモデル用
            num_classes: クラス数（model_name指定時に使用、デフォルト80）
            class_names: クラス名リスト（Noneの場合はCOCOクラス）
            conf_thresh: 信頼度閾値
            nms_thresh: NMS閾値
            input_size: 入力サイズ
            device: デバイス ("auto", "cuda", "cpu")
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        
        # デバイス設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # モデル設定を決定
        if exp_path:
            # カスタム実験ファイルを使用
            print(f"Loading experiment from: {exp_path}")
            self.exp = get_exp(exp_path, None)
            self.num_classes = self.exp.num_classes
        elif model_name:
            # 標準YOLOXモデルを使用
            if model_name not in YOLOX_MODELS:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(YOLOX_MODELS.keys())}")
            
            print(f"Using standard model: {model_name}")
            self.exp = get_exp(None, model_name)
            
            # クラス数を設定
            if num_classes:
                self.exp.num_classes = num_classes
                self.num_classes = num_classes
            else:
                self.num_classes = 80  # COCOデフォルト
                self.exp.num_classes = 80
            
            # 入力サイズを設定
            self.exp.test_size = input_size
        else:
            raise ValueError("Either exp_path or model_name must be provided")
        
        # クラス名を設定
        if class_names:
            self.class_names = class_names
        elif self.num_classes == 2:
            self.class_names = SEAL_CLASSES
        elif self.num_classes == 80:
            self.class_names = COCO_CLASSES
        else:
            # 汎用クラス名
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        
        print(f"Number of classes: {self.num_classes}")
        
        # モデル構築
        self.model = self.exp.get_model()
        self.model.eval()
        
        # 重み読み込み
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        
        # チェックポイントの形式に対応
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        
        self.model.to(self.device)
        
        # FP16モード（GPU使用時）
        if self.device.type == "cuda":
            self.model.half()
            self.fp16 = True
        else:
            self.fp16 = False
        
        # 前処理
        self.preproc = ValTransform(legacy=False)
        
        # 色を生成
        self.colors = generate_colors(self.num_classes)
        
        print("Model loaded successfully!")
    
    def detect(self, frame: np.ndarray) -> list:
        """
        フレームから物体を検出
        
        Args:
            frame: BGR画像 (numpy array)
            
        Returns:
            検出結果のリスト
        """
        height, width = frame.shape[:2]
        
        # 前処理
        img, _ = self.preproc(frame, None, self.input_size)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        if self.fp16:
            img = img.half()
        else:
            img = img.float()
        
        # 推論
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.conf_thresh,
                self.nms_thresh
            )
        
        return self._parse_outputs(outputs[0], height, width)
    
    def _parse_outputs(self, output, img_height: int, img_width: int) -> list:
        """出力をパース"""
        results = []
        
        if output is None:
            return results
        
        # スケール比率を計算
        ratio = min(
            self.input_size[0] / img_height,
            self.input_size[1] / img_width
        )
        
        output = output.cpu().numpy()
        bboxes = output[:, 0:4] / ratio
        scores = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6].astype(int)
        
        for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
            x1, y1, x2, y2 = bbox.astype(int)
            
            # 画像範囲内に収める
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # クラス名を取得
            if cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = f"class_{cls_id}"
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'class_id': int(cls_id),
                'class_name': class_name
            })
        
        return results
    
    def draw_results(
        self, 
        frame: np.ndarray, 
        results: list,
        show_label: bool = True,
        show_score: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """検出結果を描画"""
        frame = frame.copy()
        
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            cls_id = r['class_id']
            cls_name = r['class_name']
            score = r['score']
            
            # 色を取得
            color = self.colors.get(cls_id, (0, 255, 0))
            
            # バウンディングボックス
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if show_label:
                # ラベル作成
                if show_score:
                    label = f"{cls_name}: {score:.2f}"
                else:
                    label = cls_name
                
                # ラベル背景
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # ラベルテキスト
                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return frame


def process_video(
    detector: YOLOXDetector,
    video_path: str,
    output_path: str = None,
    show: bool = False,
    filter_classes: list = None
):
    """
    動画を処理
    
    Args:
        detector: YOLOXDetectorインスタンス
        video_path: 入力動画パス
        output_path: 出力動画パス (Noneの場合は保存しない)
        show: 表示するかどうか
        filter_classes: フィルタするクラス名リスト（Noneの場合は全クラス）
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # 動画情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    if filter_classes:
        print(f"Filtering classes: {filter_classes}")
    
    # 出力設定
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 統計情報
    frame_count = 0
    total_time = 0
    detection_counts = {}
    
    print("Processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 検出
        start_time = time.time()
        results = detector.detect(frame)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # フィルタ適用
        if filter_classes:
            results = [r for r in results if r['class_name'] in filter_classes]
        
        # 統計更新
        for r in results:
            cls_name = r['class_name']
            detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
        
        # 描画
        frame = detector.draw_results(frame, results)
        
        # FPS表示
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 進捗表示
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            avg_fps = frame_count / total_time
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Avg FPS: {avg_fps:.1f}")
        
        # 出力
        if writer:
            writer.write(frame)
        
        if show:
            cv2.imshow('YOLOX Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user")
                break
    
    # クリーンアップ
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("Detection Summary")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    
    if detection_counts:
        print(f"\nDetections by class:")
        for name, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
            print(f"  - {name}: {count}")
    else:
        print("\nNo detections")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")


def process_image(
    detector: YOLOXDetector,
    image_path: str,
    output_path: str = None,
    show: bool = False,
    filter_classes: list = None
):
    """
    画像を処理
    
    Args:
        detector: YOLOXDetectorインスタンス
        image_path: 入力画像パス
        output_path: 出力画像パス (Noneの場合は保存しない)
        show: 表示するかどうか
        filter_classes: フィルタするクラス名リスト
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"Image: {image_path}")
    print(f"Size: {frame.shape[1]}x{frame.shape[0]}")
    
    # 検出
    start_time = time.time()
    results = detector.detect(frame)
    elapsed = time.time() - start_time
    
    # フィルタ適用
    if filter_classes:
        results = [r for r in results if r['class_name'] in filter_classes]
    
    print(f"Detection time: {elapsed*1000:.1f}ms")
    print(f"Found {len(results)} objects")
    
    # 結果表示
    if results:
        print("\nDetections:")
        for i, r in enumerate(results, 1):
            x1, y1, x2, y2 = r['bbox']
            print(f"  {i}. {r['class_name']}: [{x1},{y1},{x2},{y2}] (score: {r['score']:.3f})")
    
    # 描画
    result_frame = detector.draw_results(frame, results)
    
    # 保存
    if output_path:
        cv2.imwrite(output_path, result_frame)
        print(f"\nOutput saved to: {output_path}")
    
    # 表示
    if show:
        cv2.imshow('YOLOX Detection', result_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def load_class_names(filepath: str) -> list:
    """クラス名ファイルを読み込み"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="YOLOX Inference Script (Extended)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # シール検出（カスタムモデル）
  python yolox_inference.py -c seal_model.pth -e seal_exp.py -i input.mp4 -o output.mp4

  # COCO事前学習モデルで推論
  python yolox_inference.py -c yolox_s.pth -m yolox_s -i input.mp4 -o output.mp4

  # 特定クラスのみ検出
  python yolox_inference.py -c yolox_s.pth -m yolox_s -i input.mp4 --filter person,car,truck

  # 画像から検出
  python yolox_inference.py -c yolox_s.pth -m yolox_s -i image.jpg -o result.jpg

Available models: yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l, yolox_x
        """
    )
    
    # モデル設定
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "-c", "--ckpt",
        type=str,
        required=True,
        help="Checkpoint file path"
    )
    model_group.add_argument(
        "-e", "--exp",
        type=str,
        default=None,
        help="Experiment file path (for custom models)"
    )
    model_group.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        choices=list(YOLOX_MODELS.keys()),
        help="Standard YOLOX model name (for pretrained models)"
    )
    model_group.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (default: 80 for COCO, 2 for seal detection)"
    )
    model_group.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Path to class names file (one class per line)"
    )
    
    # 入出力
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input video or image path"
    )
    io_group.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path"
    )
    
    # 検出パラメータ
    detect_group = parser.add_argument_group('Detection Parameters')
    detect_group.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    detect_group.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS threshold (default: 0.45)"
    )
    detect_group.add_argument(
        "--size",
        type=int,
        default=640,
        help="Input size (default: 640)"
    )
    detect_group.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter specific classes (comma-separated, e.g., 'person,car,truck')"
    )
    
    # その他
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detection results (requires GUI)"
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List available COCO classes and exit"
    )
    
    args = parser.parse_args()
    
    # COCOクラス一覧表示
    if args.list_classes:
        print("COCO Classes (80):")
        print("=" * 50)
        for i, name in enumerate(COCO_CLASSES):
            print(f"  {i:2d}: {name}")
        return
    
    # モデル設定の検証
    if not args.exp and not args.model:
        parser.error("Either --exp or --model must be specified")
    
    # クラス名読み込み
    class_names = None
    if args.class_names:
        class_names = load_class_names(args.class_names)
        print(f"Loaded {len(class_names)} class names from {args.class_names}")
    
    # フィルタクラス
    filter_classes = None
    if args.filter:
        filter_classes = [c.strip() for c in args.filter.split(',')]
    
    # 検出器初期化
    detector = YOLOXDetector(
        ckpt_path=args.ckpt,
        exp_path=args.exp,
        model_name=args.model,
        num_classes=args.num_classes,
        class_names=class_names,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=(args.size, args.size),
        device=args.device
    )
    
    # 入力ファイルの種類を判定
    input_path = Path(args.input)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    ext = input_path.suffix.lower()
    
    if ext in image_extensions:
        process_image(
            detector=detector,
            image_path=args.input,
            output_path=args.output,
            show=args.show,
            filter_classes=filter_classes
        )
    elif ext in video_extensions:
        process_video(
            detector=detector,
            video_path=args.input,
            output_path=args.output,
            show=args.show,
            filter_classes=filter_classes
        )
    else:
        print(f"Unknown file type: {ext}")
        print(f"Supported images: {image_extensions}")
        print(f"Supported videos: {video_extensions}")


if __name__ == "__main__":
    main()
