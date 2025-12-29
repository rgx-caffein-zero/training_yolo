#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX Seal Detection - Video Inference Script

YOLOXモデルを使用して動画からシールを検出するスクリプト
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


class YOLOXSealDetector:
    """YOLOXベースのシール検出クラス"""
    
    CLASS_NAMES = ['blue_triangle', 'yellow_octagon']
    CLASS_COLORS = {
        'blue_triangle': (255, 0, 0),    # BGR: 青
        'yellow_octagon': (0, 255, 255)  # BGR: 黄
    }
    
    def __init__(
        self,
        exp_path: str,
        ckpt_path: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
        device: str = "auto"
    ):
        """
        Args:
            exp_path: 実験設定ファイルのパス
            ckpt_path: チェックポイントファイルのパス
            conf_thresh: 信頼度閾値
            nms_thresh: NMS閾値
            device: デバイス ("auto", "cuda", "cpu")
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # デバイス設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # モデル読み込み
        self.exp = get_exp(exp_path, None)
        self.model = self.exp.get_model()
        self.model.eval()
        
        # 重み読み込み
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        
        # FP16モード（GPU使用時）
        if self.device.type == "cuda":
            self.model.half()
            self.fp16 = True
        else:
            self.fp16 = False
        
        # 前処理
        self.preproc = ValTransform(legacy=False)
        self.input_size = self.exp.test_size
    
    def detect(self, frame: np.ndarray) -> list:
        """
        フレームからシールを検出
        
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
                self.exp.num_classes,
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
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'class_id': int(cls_id),
                'class_name': self.CLASS_NAMES[cls_id]
            })
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: list) -> np.ndarray:
        """検出結果を描画"""
        frame = frame.copy()
        
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            cls_name = r['class_name']
            score = r['score']
            color = self.CLASS_COLORS[cls_name]
            
            # バウンディングボックス
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ラベル背景
            label = f"{cls_name}: {score:.2f}"
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
    detector: YOLOXSealDetector,
    video_path: str,
    output_path: str = None,
    show: bool = False
):
    """
    動画を処理
    
    Args:
        detector: YOLOXSealDetectorインスタンス
        video_path: 入力動画パス
        output_path: 出力動画パス (Noneの場合は保存しない)
        show: 表示するかどうか
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
    
    # 出力設定
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 統計情報
    frame_count = 0
    total_time = 0
    detection_counts = {name: 0 for name in YOLOXSealDetector.CLASS_NAMES}
    
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
        
        # 統計更新
        for r in results:
            detection_counts[r['class_name']] += 1
        
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
            cv2.imshow('YOLOX Seal Detection', frame)
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
    print(f"Detections:")
    for name, count in detection_counts.items():
        print(f"  - {name}: {count}")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOX Seal Detection Video Inference")
    
    parser.add_argument(
        "-e", "--exp",
        type=str,
        default="/workspace/exps/custom/seal_detection_exp.py",
        help="Experiment file path"
    )
    parser.add_argument(
        "-c", "--ckpt",
        type=str,
        required=True,
        help="Checkpoint file path"
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output video path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detection results"
    )
    
    args = parser.parse_args()
    
    # 検出器初期化
    detector = YOLOXSealDetector(
        exp_path=args.exp,
        ckpt_path=args.ckpt,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        device=args.device
    )
    
    # 動画処理
    process_video(
        detector=detector,
        video_path=args.video,
        output_path=args.output,
        show=args.show
    )


if __name__ == "__main__":
    main()
