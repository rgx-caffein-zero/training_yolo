#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Detection Script

動画からシールを検出するスクリプト
"""

import argparse
import time
from pathlib import Path

import cv2

from color_shape_detector import ColorShapeDetector, Detection


def process_video(
    video_path: str,
    output_path: str = None,
    config_path: str = None,
    show: bool = False,
    draw_contour: bool = True,
    draw_vertices: bool = False
):
    """
    動画を処理してシールを検出
    
    Args:
        video_path: 入力動画パス
        output_path: 出力動画パス（Noneの場合は保存しない）
        config_path: 設定ファイルパス
        show: 表示するかどうか
        draw_contour: 輪郭を描画するか
        draw_vertices: 頂点数を表示するか
    """
    # 検出器初期化
    detector = ColorShapeDetector(config_path)
    
    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # 動画情報
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
    
    # 統計
    frame_count = 0
    total_time = 0
    detection_counts = {'blue_triangle': 0, 'yellow_octagon': 0}
    
    print("Processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 検出
        start_time = time.time()
        detections = detector.detect(frame)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # 統計更新
        for det in detections:
            if det.class_name in detection_counts:
                detection_counts[det.class_name] += 1
        
        # 描画
        result = detector.draw_detections(
            frame, detections, 
            draw_contour=False,
            draw_hull=draw_contour,
            draw_vertices=draw_vertices
        )
        
        # FPS表示
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(
            result, f"FPS: {current_fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # 検出数表示
        y_offset = 60
        for class_name, count in detection_counts.items():
            cv2.putText(
                result, f"{class_name}: {count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        # 進捗表示
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Avg FPS: {avg_fps:.1f}")
        
        # 出力
        if writer:
            writer.write(result)
        
        if show:
            cv2.imshow('Seal Detection', result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Interrupted by user")
                break
            elif key == ord('d'):  # デバッグ表示切替
                draw_vertices = not draw_vertices
    
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
    print(f"Average FPS: {frame_count / total_time:.2f}" if total_time > 0 else "N/A")
    print(f"Detections:")
    for name, count in detection_counts.items():
        print(f"  - {name}: {count}")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")


def process_image(
    image_path: str,
    output_path: str = None,
    config_path: str = None,
    show: bool = True,
    draw_contour: bool = True,
    draw_vertices: bool = True
):
    """
    画像からシールを検出
    
    Args:
        image_path: 入力画像パス
        output_path: 出力画像パス
        config_path: 設定ファイルパス
        show: 表示するかどうか
        draw_contour: 輪郭を描画するか
        draw_vertices: 頂点数を表示するか
    """
    detector = ColorShapeDetector(config_path)
    
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"Image: {image_path}")
    print(f"Size: {frame.shape[1]}x{frame.shape[0]}")
    
    # 検出
    start_time = time.time()
    detections = detector.detect(frame)
    elapsed = time.time() - start_time
    
    print(f"Detection time: {elapsed*1000:.1f}ms")
    print(f"Found {len(detections)} objects:")
    
    for det in detections:
        print(f"  - {det.class_name}: bbox={det.bbox}, vertices={det.vertices}, conf={det.confidence:.2f}")
    
    # 描画
    result = detector.draw_detections(
        frame, detections,
        draw_contour=False,
        draw_hull=draw_contour,
        draw_vertices=draw_vertices
    )
    
    # 保存
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Output saved to: {output_path}")
    
    # 表示
    if show:
        # デバッグ用マスク表示
        masks = detector.get_debug_masks(frame)
        
        cv2.imshow('Detection Result', result)
        cv2.imshow('Blue Mask', masks['blue'])
        cv2.imshow('Yellow Mask', masks['yellow'])
        
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Seal Detection using Color and Shape")
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input video or image path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Config YAML file path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detection results"
    )
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="Don't draw contours"
    )
    parser.add_argument(
        "--vertices",
        action="store_true",
        help="Show vertex count"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 画像か動画かを拡張子で判定
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    ext = input_path.suffix.lower()
    
    if ext in image_extensions:
        process_image(
            image_path=args.input,
            output_path=args.output,
            config_path=args.config,
            show=args.show,
            draw_contour=not args.no_contour,
            draw_vertices=args.vertices
        )
    elif ext in video_extensions:
        process_video(
            video_path=args.input,
            output_path=args.output,
            config_path=args.config,
            show=args.show,
            draw_contour=not args.no_contour,
            draw_vertices=args.vertices
        )
    else:
        print(f"Unknown file type: {ext}")
        print(f"Supported images: {image_extensions}")
        print(f"Supported videos: {video_extensions}")


if __name__ == "__main__":
    main()
