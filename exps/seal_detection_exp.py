#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX Experiment Configuration for Seal Detection

シール検出用のカスタム設定ファイル
- 2クラス: blue_triangle (青い三角形), yellow_octagon (黄色い八角形)
- 環境変化に強いデータ拡張設定
"""

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        
        # ============ Model Config ============
        # YOLOX-S相当の設定（バランス重視）
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 2  # blue_triangle, yellow_octagon
        
        # 入力サイズ
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (14, 26)  # マルチスケール学習用
        
        # ============ Training Config ============
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15  # 最後の15エポックはデータ拡張なし
        self.min_lr_ratio = 0.05
        self.ema = True
        
        # バッチサイズ
        self.data_num_workers = 4
        self.eval_interval = 5
        
        # ============ Data Augmentation ============
        # 環境変化に強くするための拡張設定
        
        # Mosaic拡張（複数画像の合成）
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.5, 1.5)
        
        # MixUp拡張（画像のブレンド）
        self.mixup_prob = 0.5
        self.mixup_scale = (0.5, 1.5)
        
        # HSV拡張（照明変化対策）
        self.hsv_prob = 1.0
        self.hsv_h = 0.015  # 色相
        self.hsv_s = 0.7    # 彩度（大きめで環境変化に対応）
        self.hsv_v = 0.4    # 明度（照明変化対策）
        
        # 幾何学的変換
        self.degrees = 15.0      # 回転角度
        self.translate = 0.1     # 平行移動
        self.scale = (0.5, 1.5)  # スケール変動
        self.shear = 2.0         # せん断
        self.flip_prob = 0.5     # 水平反転
        
        # ============ Loss Config ============
        self.use_l1 = False
        
        # ============ Dataset Config ============
        self.data_dir = "/workspace/dataset"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.train_name = "train"
        self.val_name = "val"
        
        # ============ Output Config ============
        self.output_dir = "/workspace/outputs"
        self.exp_name = "seal_detection"
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """カスタムデータローダーの設定"""
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
            enable_mixup=not no_aug,
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
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """評価用データローダー"""
        from yolox.data import COCODataset, ValTransform
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_name if not testdev else self.test_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """評価器の設定"""
        from yolox.evaluators import COCOEvaluator
        
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
