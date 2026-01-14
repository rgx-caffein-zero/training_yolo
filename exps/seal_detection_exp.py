#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seal Detection Experiment Configuration for YOLOX

シール検出用のYOLOX実験設定
- データセットパスはYOLOX外の /workspace/dataset を参照
- 2クラス検出: blue_triangle (冷蔵), yellow_octagon (冷凍)
"""

import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        
        # ============================================
        # データセット設定（YOLOX COCODataset形式）
        # ============================================
        # YOLOX外のデータセットパスを指定
        # Dockerコンテナ内: /workspace/dataset
        # ホスト: ~/training_yolo/dataset
        self.data_dir = "/workspace/dataset"
        
        # アノテーションファイル名（annotations/ディレクトリ内）
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        
        # 画像ディレクトリ名
        # /workspace/dataset/train2017/, /workspace/dataset/val2017/
        self.train_name = "train2017"
        self.val_name = "val2017"
        
        # ============================================
        # モデル設定
        # ============================================
        # クラス数（blue_triangle, yellow_octagon）
        self.num_classes = 2
        
        # 実験名（出力ディレクトリ名）
        self.exp_name = "seal_detection"
        
        # ベースモデル: YOLOX-S
        self.depth = 0.33
        self.width = 0.50
        
        # ============================================
        # 学習設定
        # ============================================
        # 入力サイズ
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        
        # バッチサイズ（GPU メモリに応じて調整）
        self.data_num_workers = 4
        
        # 学習エポック数
        self.max_epoch = 100
        
        # ウォームアップ
        self.warmup_epochs = 5
        
        # 学習率
        self.basic_lr_per_img = 0.01 / 64.0
        
        # ============================================
        # データ拡張設定
        # ============================================
        # Mosaic拡張
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.5, 1.5)
        
        # MixUp拡張
        self.mixup_prob = 0.5
        self.mixup_scale = (0.5, 1.5)
        
        # HSV拡張（色の変化に対応）
        self.hsv_prob = 1.0
        
        # フリップ拡張
        self.flip_prob = 0.5
        
        # ============================================
        # 評価設定
        # ============================================
        # NMS閾値
        self.nmsthre = 0.45
        
        # 信頼度閾値（評価時）
        self.test_conf = 0.01
        
        # ============================================
        # 出力設定
        # ============================================
        # 出力ディレクトリ
        self.output_dir = "/workspace/outputs"
        
        # チェックポイント保存間隔
        self.save_history_ckpt = True
        
        # 評価間隔
        self.eval_interval = 5
        
        # プリント間隔
        self.print_interval = 10

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """データローダーを取得"""
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache_img,
                name=self.train_name,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """評価用データローダーを取得"""
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.val_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """評価器を取得"""
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
