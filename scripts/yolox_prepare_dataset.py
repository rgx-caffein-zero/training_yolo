#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Utilities for YOLOX Seal Detection

動画からのフレーム抽出とCOCO形式アノテーション作成のユーティリティ
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    interval: int = 30,
    max_frames: int = None
) -> list:
    """
    動画からフレームを抽出
    
    Args:
        video_path: 入力動画パス
        output_dir: 出力ディレクトリ
        interval: フレーム抽出間隔
        max_frames: 最大フレーム数
        
    Returns:
        抽出したファイルパスのリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Extracting every {interval} frames...")
    
    extracted_files = []
    frame_count = 0
    saved_count = 0
    
    video_name = Path(video_path).stem
    
    pbar = tqdm(total=total_frames, desc="Extracting")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            filename = f"{video_name}_{frame_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            extracted_files.append(filepath)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames to {output_dir}")
    
    return extracted_files


def create_empty_coco_annotation(
    image_dir: str,
    output_path: str,
    categories: list = None
) -> dict:
    """
    空のCOCO形式アノテーションファイルを作成
    
    Args:
        image_dir: 画像ディレクトリ
        output_path: 出力JSONパス
        categories: カテゴリリスト
        
    Returns:
        COCOアノテーション辞書
    """
    if categories is None:
        categories = [
            {"id": 1, "name": "blue_triangle", "supercategory": "seal"},
            {"id": 2, "name": "yellow_octagon", "supercategory": "seal"}
        ]
    
    # 画像情報収集
    images = []
    image_id = 1
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for filename in sorted(os.listdir(image_dir)):
        if Path(filename).suffix.lower() not in image_extensions:
            continue
        
        filepath = os.path.join(image_dir, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"Warning: Cannot read {filepath}")
            continue
        
        height, width = img.shape[:2]
        
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        image_id += 1
    
    # COCOアノテーション構造
    coco_annotation = {
        "info": {
            "description": "Seal Detection Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": images,
        "annotations": [],  # 空（後でアノテーションツールで追加）
        "categories": categories
    }
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_annotation, f, indent=2, ensure_ascii=False)
    
    print(f"Created COCO annotation file: {output_path}")
    print(f"  - Images: {len(images)}")
    print(f"  - Categories: {[c['name'] for c in categories]}")
    
    return coco_annotation


def split_dataset(
    image_dir: str,
    annotation_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    データセットを学習用と検証用に分割
    
    Args:
        image_dir: 画像ディレクトリ
        annotation_path: アノテーションJSONパス
        output_dir: 出力ディレクトリ
        train_ratio: 学習データの割合
        seed: ランダムシード
    """
    random.seed(seed)
    
    # アノテーション読み込み
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # 画像IDでアノテーションをグループ化
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    
    # シャッフルして分割
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # 出力ディレクトリ作成
    train_img_dir = os.path.join(output_dir, 'train')
    val_img_dir = os.path.join(output_dir, 'val')
    ann_dir = os.path.join(output_dir, 'annotations')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    
    # 画像コピーとアノテーション分割
    def process_split(images_list, img_dir, split_name):
        new_images = []
        new_annotations = []
        new_ann_id = 1
        
        for new_img_id, img in enumerate(tqdm(images_list, desc=f"Processing {split_name}"), 1):
            # 画像コピー
            src_path = os.path.join(image_dir, img['file_name'])
            dst_path = os.path.join(img_dir, img['file_name'])
            
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)
            
            # 画像情報更新
            new_img = img.copy()
            old_img_id = img['id']
            new_img['id'] = new_img_id
            new_images.append(new_img)
            
            # アノテーション更新
            if old_img_id in ann_by_image:
                for ann in ann_by_image[old_img_id]:
                    new_ann = ann.copy()
                    new_ann['id'] = new_ann_id
                    new_ann['image_id'] = new_img_id
                    new_annotations.append(new_ann)
                    new_ann_id += 1
        
        return new_images, new_annotations
    
    train_images, train_annotations = process_split(train_images, train_img_dir, "train")
    val_images, val_annotations = process_split(val_images, val_img_dir, "val")
    
    # アノテーションファイル作成
    for split_name, split_images, split_annotations in [
        ("train", train_images, train_annotations),
        ("val", val_images, val_annotations)
    ]:
        split_coco = {
            "info": coco_data['info'],
            "licenses": coco_data.get('licenses', []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco_data['categories']
        }
        
        output_path = os.path.join(ann_dir, f"instances_{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_coco, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name}: {len(split_images)} images, {len(split_annotations)} annotations")
    
    print(f"\nDataset split completed!")
    print(f"Output directory: {output_dir}")


def verify_dataset(dataset_dir: str):
    """
    データセットの整合性を検証
    
    Args:
        dataset_dir: データセットディレクトリ
    """
    print("Verifying dataset...")
    
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_dir, split)
        ann_path = os.path.join(dataset_dir, 'annotations', f'instances_{split}.json')
        
        if not os.path.exists(ann_path):
            print(f"Warning: Annotation file not found: {ann_path}")
            continue
        
        with open(ann_path, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data['images']
        annotations = coco_data['annotations']
        
        # 画像ファイル存在確認
        missing_images = []
        for img in images:
            img_path = os.path.join(img_dir, img['file_name'])
            if not os.path.exists(img_path):
                missing_images.append(img['file_name'])
        
        # 統計
        ann_count_by_cat = {}
        for ann in annotations:
            cat_id = ann['category_id']
            ann_count_by_cat[cat_id] = ann_count_by_cat.get(cat_id, 0) + 1
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {len(images)}")
        print(f"  Annotations: {len(annotations)}")
        print(f"  Missing images: {len(missing_images)}")
        
        if missing_images:
            print(f"    Missing: {missing_images[:5]}...")
        
        print(f"  Annotations by category:")
        for cat in coco_data['categories']:
            count = ann_count_by_cat.get(cat['id'], 0)
            print(f"    - {cat['name']}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation Utilities for YOLOX")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # extract コマンド
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('-v', '--video', required=True, help='Input video path')
    extract_parser.add_argument('-o', '--output', required=True, help='Output directory')
    extract_parser.add_argument('-i', '--interval', type=int, default=30, help='Frame interval')
    extract_parser.add_argument('-m', '--max', type=int, default=None, help='Max frames')
    
    # create コマンド
    create_parser = subparsers.add_parser('create', help='Create empty COCO annotation')
    create_parser.add_argument('-i', '--images', required=True, help='Image directory')
    create_parser.add_argument('-o', '--output', required=True, help='Output JSON path')
    
    # split コマンド
    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('-i', '--images', required=True, help='Image directory')
    split_parser.add_argument('-a', '--annotation', required=True, help='Annotation JSON path')
    split_parser.add_argument('-o', '--output', required=True, help='Output directory')
    split_parser.add_argument('-r', '--ratio', type=float, default=0.8, help='Train ratio')
    
    # verify コマンド
    verify_parser = subparsers.add_parser('verify', help='Verify dataset')
    verify_parser.add_argument('-d', '--dataset', required=True, help='Dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_frames(args.video, args.output, args.interval, args.max)
    elif args.command == 'create':
        create_empty_coco_annotation(args.images, args.output)
    elif args.command == 'split':
        split_dataset(args.images, args.annotation, args.output, args.ratio)
    elif args.command == 'verify':
        verify_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
