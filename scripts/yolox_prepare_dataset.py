#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Utilities for YOLOX Seal Detection

動画からのフレーム抽出とCOCO形式アノテーション作成のユーティリティ

出力構造（YOLOX COCODataset形式）:
    dataset/
    ├── train2017/
    │   └── *.jpg
    ├── val2017/
    │   └── *.jpg
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
"""

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm


# デフォルトカテゴリ（シール検出用）
DEFAULT_CATEGORIES = [
    {"id": 1, "name": "blue_triangle", "supercategory": "seal"},
    {"id": 2, "name": "yellow_octagon", "supercategory": "seal"}
]


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
        categories = DEFAULT_CATEGORIES
    
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
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
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
    データセットを学習用と検証用に分割（YOLOX COCODataset形式）
    
    Args:
        image_dir: 画像ディレクトリ
        annotation_path: アノテーションJSONパス
        output_dir: 出力ディレクトリ
        train_ratio: 学習データの割合
        seed: ランダムシード
    
    出力構造:
        output_dir/
        ├── train2017/
        │   └── *.jpg
        ├── val2017/
        │   └── *.jpg
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
    """
    random.seed(seed)
    
    # アノテーション読み込み
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    print(f"Loaded {len(images)} images, {len(annotations)} annotations")
    
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
    
    # 出力ディレクトリ作成（YOLOX COCODataset形式）
    train_img_dir = os.path.join(output_dir, 'train2017')
    val_img_dir = os.path.join(output_dir, 'val2017')
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
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
                continue
            
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
    
    train_images, train_annotations = process_split(train_images, train_img_dir, "train2017")
    val_images, val_annotations = process_split(val_images, val_img_dir, "val2017")
    
    # アノテーションファイル作成（YOLOX COCODataset形式）
    for split_name, split_images, split_annotations in [
        ("train2017", train_images, train_annotations),
        ("val2017", val_images, val_annotations)
    ]:
        split_coco = {
            "info": coco_data.get('info', {}),
            "licenses": coco_data.get('licenses', []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco_data['categories']
        }
        
        # instances_train2017.json / instances_val2017.json
        output_path = os.path.join(ann_dir, f"instances_{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_coco, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name}: {len(split_images)} images, {len(split_annotations)} annotations")
    
    print(f"\nDataset split completed!")
    print(f"Output directory: {output_dir}")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    print(f"  ├── train2017/          ({len(train_images)} images)")
    print(f"  ├── val2017/            ({len(val_images)} images)")
    print(f"  └── annotations/")
    print(f"      ├── instances_train2017.json")
    print(f"      └── instances_val2017.json")


def convert_cvat_coco(
    cvat_annotation_path: str,
    output_path: str = None
):
    """
    CVATからエクスポートしたCOCO形式アノテーションを確認・変換
    
    CVATのCOCO 1.0エクスポートは基本的にそのまま使用可能ですが、
    カテゴリIDの確認と必要に応じた調整を行います。
    
    Args:
        cvat_annotation_path: CVATからエクスポートしたJSONパス
        output_path: 出力パス（Noneの場合は上書き保存しない）
    """
    with open(cvat_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    print("=" * 50)
    print("CVAT COCO Annotation Summary")
    print("=" * 50)
    
    # カテゴリ確認
    print("\nCategories:")
    for cat in coco_data.get('categories', []):
        print(f"  ID {cat['id']}: {cat['name']}")
    
    # 画像数
    images = coco_data.get('images', [])
    print(f"\nImages: {len(images)}")
    
    # アノテーション数
    annotations = coco_data.get('annotations', [])
    print(f"Annotations: {len(annotations)}")
    
    # カテゴリ別アノテーション数
    ann_by_cat = {}
    for ann in annotations:
        cat_id = ann.get('category_id')
        ann_by_cat[cat_id] = ann_by_cat.get(cat_id, 0) + 1
    
    print("\nAnnotations by category:")
    for cat in coco_data.get('categories', []):
        count = ann_by_cat.get(cat['id'], 0)
        print(f"  {cat['name']}: {count}")
    
    # bbox形式の確認
    if annotations:
        sample_ann = annotations[0]
        print(f"\nBbox format (sample): {sample_ann.get('bbox')}")
        print("  (COCO format: [x, y, width, height])")
    
    # 出力
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {output_path}")
    
    print("=" * 50)
    
    return coco_data


def verify_dataset(dataset_dir: str):
    """
    データセットの整合性を検証（YOLOX COCODataset形式）
    
    Args:
        dataset_dir: データセットディレクトリ
    """
    print("=" * 50)
    print("Verifying YOLOX Dataset")
    print("=" * 50)
    print(f"Dataset directory: {dataset_dir}\n")
    
    # ディレクトリ構造確認
    expected_structure = {
        'train2017': 'Training images',
        'val2017': 'Validation images',
        'annotations': 'Annotation files'
    }
    
    print("Directory structure:")
    for dirname, description in expected_structure.items():
        dirpath = os.path.join(dataset_dir, dirname)
        exists = os.path.isdir(dirpath)
        status = "✓" if exists else "✗"
        print(f"  {status} {dirname}/ - {description}")
    
    print()
    
    # 各splitを検証
    for split in ['train2017', 'val2017']:
        img_dir = os.path.join(dataset_dir, split)
        ann_path = os.path.join(dataset_dir, 'annotations', f'instances_{split}.json')
        
        print(f"{split.upper()}:")
        
        # アノテーションファイル確認
        if not os.path.exists(ann_path):
            print(f"  ✗ Annotation file not found: {ann_path}")
            continue
        
        print(f"  ✓ Annotation file: instances_{split}.json")
        
        with open(ann_path, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data.get('images', [])
        annotations = coco_data.get('annotations', [])
        categories = coco_data.get('categories', [])
        
        # 画像ファイル存在確認
        missing_images = []
        existing_images = 0
        
        for img in images:
            img_path = os.path.join(img_dir, img['file_name'])
            if os.path.exists(img_path):
                existing_images += 1
            else:
                missing_images.append(img['file_name'])
        
        print(f"  Images: {existing_images}/{len(images)}")
        
        if missing_images:
            print(f"  ✗ Missing images: {len(missing_images)}")
            for name in missing_images[:3]:
                print(f"      - {name}")
            if len(missing_images) > 3:
                print(f"      ... and {len(missing_images) - 3} more")
        
        # アノテーション統計
        ann_count_by_cat = {}
        for ann in annotations:
            cat_id = ann.get('category_id')
            ann_count_by_cat[cat_id] = ann_count_by_cat.get(cat_id, 0) + 1
        
        print(f"  Annotations: {len(annotations)}")
        for cat in categories:
            count = ann_count_by_cat.get(cat['id'], 0)
            print(f"    - {cat['name']}: {count}")
        
        print()
    
    # 推奨事項
    print("=" * 50)
    print("Recommendations:")
    print("=" * 50)
    
    # 各クラスのアノテーション数を確認
    train_ann_path = os.path.join(dataset_dir, 'annotations', 'instances_train2017.json')
    if os.path.exists(train_ann_path):
        with open(train_ann_path, 'r') as f:
            train_data = json.load(f)
        
        ann_counts = {}
        for ann in train_data.get('annotations', []):
            cat_id = ann.get('category_id')
            ann_counts[cat_id] = ann_counts.get(cat_id, 0) + 1
        
        for cat in train_data.get('categories', []):
            count = ann_counts.get(cat['id'], 0)
            if count < 50:
                print(f"  ⚠ '{cat['name']}' has only {count} annotations (recommended: 50+)")
            elif count < 100:
                print(f"  △ '{cat['name']}' has {count} annotations (good: 100+)")
            else:
                print(f"  ✓ '{cat['name']}' has {count} annotations")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Preparation Utilities for YOLOX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 動画からフレーム抽出
  python yolox_prepare_dataset.py extract -v input.mp4 -o raw_images -i 30

  # 空のアノテーションファイル作成（CVATにインポート用）
  python yolox_prepare_dataset.py create -i raw_images -o annotations.json

  # CVATアノテーションの確認
  python yolox_prepare_dataset.py check -a cvat_annotations.json

  # データセット分割（YOLOX形式）
  python yolox_prepare_dataset.py split -i raw_images -a annotations.json -o dataset

  # データセット検証
  python yolox_prepare_dataset.py verify -d dataset

Output structure (YOLOX COCODataset format):
  dataset/
  ├── train2017/
  │   └── *.jpg
  ├── val2017/
  │   └── *.jpg
  └── annotations/
      ├── instances_train2017.json
      └── instances_val2017.json
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # extract コマンド
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('-v', '--video', required=True, help='Input video path')
    extract_parser.add_argument('-o', '--output', required=True, help='Output directory')
    extract_parser.add_argument('-i', '--interval', type=int, default=30, help='Frame interval (default: 30)')
    extract_parser.add_argument('-m', '--max', type=int, default=None, help='Max frames to extract')
    
    # create コマンド
    create_parser = subparsers.add_parser('create', help='Create empty COCO annotation file')
    create_parser.add_argument('-i', '--images', required=True, help='Image directory')
    create_parser.add_argument('-o', '--output', required=True, help='Output JSON path')
    
    # check コマンド（CVATアノテーション確認）
    check_parser = subparsers.add_parser('check', help='Check CVAT COCO annotation')
    check_parser.add_argument('-a', '--annotation', required=True, help='CVAT annotation JSON path')
    check_parser.add_argument('-o', '--output', default=None, help='Output path (optional)')
    
    # split コマンド
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('-i', '--images', required=True, help='Image directory')
    split_parser.add_argument('-a', '--annotation', required=True, help='Annotation JSON path')
    split_parser.add_argument('-o', '--output', required=True, help='Output directory')
    split_parser.add_argument('-r', '--ratio', type=float, default=0.8, help='Train ratio (default: 0.8)')
    split_parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed (default: 42)')
    
    # verify コマンド
    verify_parser = subparsers.add_parser('verify', help='Verify dataset structure')
    verify_parser.add_argument('-d', '--dataset', required=True, help='Dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_frames(args.video, args.output, args.interval, args.max)
    elif args.command == 'create':
        create_empty_coco_annotation(args.images, args.output)
    elif args.command == 'check':
        convert_cvat_coco(args.annotation, args.output)
    elif args.command == 'split':
        split_dataset(args.images, args.annotation, args.output, args.ratio, args.seed)
    elif args.command == 'verify':
        verify_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
