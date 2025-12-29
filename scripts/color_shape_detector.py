#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color and Shape Based Seal Detector

HSV色空間での色検出と輪郭形状分析によるシール検出
- blue_triangle: 青い三角形（冷蔵シール）
- yellow_octagon: 黄色い八角形（冷凍シール）

対応機能:
- 凸包（Convex Hull）による内部白領域の無視
- モルフォロジー変換による輪郭の統合
"""

import cv2
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path


@dataclass
class Detection:
    """検出結果"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    class_name: str
    confidence: float
    contour: np.ndarray
    hull: np.ndarray  # 凸包
    vertices: int  # 検出された頂点数


class ColorShapeDetector:
    """色と形状によるシール検出器"""
    
    # デフォルトのHSV範囲
    DEFAULT_CONFIG = {
        'blue': {
            'hsv_lower': [100, 80, 80],
            'hsv_upper': [130, 255, 255],
            'expected_vertices': 3,
            'vertex_tolerance': 1,
            'class_name': 'blue_triangle'
        },
        'yellow': {
            'hsv_lower': [20, 80, 80],
            'hsv_upper': [40, 255, 255],
            'expected_vertices': 8,
            'vertex_tolerance': 2,
            'class_name': 'yellow_octagon'
        },
        'detection': {
            'min_area': 500,
            'max_area': 500000,
            'epsilon_ratio': 0.02,
            'morph_kernel_size': 5,
            'blur_kernel_size': 5,
            # 新規オプション
            'use_convex_hull': True,          # 凸包を使用
            'fill_holes': True,               # 内部の穴を埋める
            'morph_iterations': 2,            # モルフォロジー変換の反復回数
            'dilate_iterations': 3,           # 膨張処理の反復回数
        }
    }
    
    # 描画用の色（BGR）
    COLORS = {
        'blue_triangle': (255, 0, 0),    # 青
        'yellow_octagon': (0, 255, 255)  # 黄
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス（YAML）。Noneの場合はデフォルト設定
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.DEFAULT_CONFIG.copy()
        
        # モルフォロジー変換用カーネル
        kernel_size = self.config['detection']['morph_kernel_size']
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        フレームからシールを検出
        
        Args:
            frame: BGR画像
            
        Returns:
            検出結果のリスト
        """
        detections = []
        
        # 前処理
        processed = self._preprocess(frame)
        
        # 各色について検出
        for color_key in ['blue', 'yellow']:
            color_config = self.config[color_key]
            
            # 色マスク作成
            mask = self._create_color_mask(processed, color_config)
            
            # 輪郭検出
            contours = self._find_contours(mask)
            
            # 各輪郭を評価
            color_detections = []
            for contour in contours:
                detection = self._evaluate_contour(contour, color_config)
                if detection:
                    color_detections.append(detection)
            
            # 近接した検出をマージ
            merged = self._merge_nearby_detections(color_detections)
            detections.extend(merged)
        
        return detections
    
    def _merge_nearby_detections(
        self, 
        detections: List[Detection], 
        iou_threshold: float = 0.1,
        distance_threshold: float = 150
    ) -> List[Detection]:
        """
        近接した検出結果をマージ
        
        Args:
            detections: 検出結果リスト
            iou_threshold: IoU閾値
            distance_threshold: 距離閾値（ピクセル）
            
        Returns:
            マージ後の検出結果リスト
        """
        if len(detections) <= 1:
            return detections
        
        # 検出結果を面積でソート（大きい順）
        detections = sorted(detections, key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        
        merged = []
        used = [False] * len(detections)
        
        for i, det1 in enumerate(detections):
            if used[i]:
                continue
            
            # マージ候補を収集
            to_merge = [det1]
            used[i] = True
            
            for j, det2 in enumerate(detections):
                if i == j or used[j]:
                    continue
                
                # バウンディングボックスの距離またはIoUをチェック
                if self._should_merge(det1.bbox, det2.bbox, iou_threshold, distance_threshold):
                    to_merge.append(det2)
                    used[j] = True
            
            # マージ実行
            if len(to_merge) > 1:
                merged_det = self._merge_detections(to_merge)
                merged.append(merged_det)
            else:
                merged.append(det1)
        
        return merged
    
    def _should_merge(
        self, 
        bbox1: Tuple[int, int, int, int], 
        bbox2: Tuple[int, int, int, int],
        iou_threshold: float,
        distance_threshold: float
    ) -> bool:
        """2つのバウンディングボックスをマージすべきか判定"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 中心点間の距離
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        
        if distance < distance_threshold:
            return True
        
        # IoU計算
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1+w1, x2+w2)
        inter_y2 = min(y1+h1, y2+h2)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = w1 * h1
            area2 = w2 * h2
            iou = inter_area / (area1 + area2 - inter_area)
            
            if iou > iou_threshold:
                return True
        
        return False
    
    def _merge_detections(self, detections: List[Detection]) -> Detection:
        """複数の検出結果を1つにマージ"""
        # すべての輪郭を結合
        all_points = np.vstack([d.contour for d in detections])
        
        # 結合した輪郭の凸包を計算
        merged_hull = cv2.convexHull(all_points)
        
        # 新しいバウンディングボックス
        x, y, w, h = cv2.boundingRect(merged_hull)
        
        # 頂点数を再計算
        epsilon = self.config['detection']['epsilon_ratio'] * cv2.arcLength(merged_hull, True)
        approx = cv2.approxPolyDP(merged_hull, epsilon, True)
        
        # 信頼度は最大値を使用
        max_conf = max(d.confidence for d in detections)
        
        return Detection(
            bbox=(x, y, w, h),
            class_name=detections[0].class_name,
            confidence=max_conf,
            contour=all_points,
            hull=merged_hull,
            vertices=len(approx)
        )
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """前処理：ブラーとHSV変換"""
        blur_size = self.config['detection']['blur_kernel_size']
        blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv
    
    def _create_color_mask(self, hsv: np.ndarray, color_config: dict) -> np.ndarray:
        """HSV色マスクを作成（強化版）"""
        lower = np.array(color_config['hsv_lower'])
        upper = np.array(color_config['hsv_upper'])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        detection_cfg = self.config['detection']
        
        # モルフォロジー変換（強化版）
        iterations = detection_cfg.get('morph_iterations', 2)
        
        # オープニング（ノイズ除去）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=iterations)
        
        # クロージング（穴埋め）
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=iterations)
        
        # 内部の穴を埋める処理
        if detection_cfg.get('fill_holes', True):
            mask = self._fill_holes(mask)
        
        # 追加の膨張処理（輪郭を繋げる）
        dilate_iter = detection_cfg.get('dilate_iterations', 3)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, self.morph_kernel, iterations=dilate_iter)
            mask = cv2.erode(mask, self.morph_kernel, iterations=dilate_iter)
        
        return mask
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """マスク内の穴を埋める"""
        # 輪郭を検出
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 外側の輪郭を塗りつぶし
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        
        return filled
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """輪郭を検出"""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def _evaluate_contour(
        self, contour: np.ndarray, color_config: dict
    ) -> Optional[Detection]:
        """
        輪郭を評価して検出結果を返す（凸包対応版）
        
        Args:
            contour: 輪郭
            color_config: 色設定
            
        Returns:
            条件を満たせばDetection、そうでなければNone
        """
        detection_cfg = self.config['detection']
        
        # 凸包を計算
        hull = cv2.convexHull(contour)
        
        # 凸包を使用するかどうか
        use_hull = detection_cfg.get('use_convex_hull', True)
        target_contour = hull if use_hull else contour
        
        # 面積フィルタ（凸包の面積を使用）
        area = cv2.contourArea(target_contour)
        min_area = detection_cfg['min_area']
        max_area = detection_cfg['max_area']
        
        if area < min_area or area > max_area:
            return None
        
        # 輪郭近似（凸包に対して行う）
        epsilon = detection_cfg['epsilon_ratio'] * cv2.arcLength(target_contour, True)
        approx = cv2.approxPolyDP(target_contour, epsilon, True)
        vertices = len(approx)
        
        # 頂点数チェック
        expected = color_config['expected_vertices']
        tolerance = color_config['vertex_tolerance']
        
        if abs(vertices - expected) > tolerance:
            return None
        
        # 信頼度計算（頂点数の一致度 + 面積の充実度）
        vertex_score = 1.0 - (abs(vertices - expected) / (tolerance + 1))
        
        # 凸包と元輪郭の面積比（充実度）
        original_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = original_area / hull_area if hull_area > 0 else 0
        
        # 総合信頼度
        confidence = vertex_score * 0.7 + solidity * 0.3
        
        # バウンディングボックス（凸包から計算）
        x, y, w, h = cv2.boundingRect(hull)
        
        return Detection(
            bbox=(x, y, w, h),
            class_name=color_config['class_name'],
            confidence=confidence,
            contour=contour,
            hull=hull,
            vertices=vertices
        )
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        draw_contour: bool = False,
        draw_hull: bool = True,
        draw_vertices: bool = False
    ) -> np.ndarray:
        """
        検出結果を描画
        
        Args:
            frame: 描画先フレーム
            detections: 検出結果リスト
            draw_contour: 元の輪郭を描画するか
            draw_hull: 凸包を描画するか
            draw_vertices: 頂点数を表示するか
            
        Returns:
            描画済みフレーム
        """
        result = frame.copy()
        
        for det in detections:
            color = self.COLORS.get(det.class_name, (0, 255, 0))
            x, y, w, h = det.bbox
            
            # バウンディングボックス
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # 凸包
            if draw_hull and det.hull is not None:
                cv2.drawContours(result, [det.hull], -1, color, 2)
            
            # 元の輪郭
            if draw_contour:
                cv2.drawContours(result, [det.contour], -1, (0, 255, 0), 1)
            
            # ラベル
            label = f"{det.class_name}"
            if draw_vertices:
                label += f" (v={det.vertices})"
            label += f" {det.confidence:.2f}"
            
            # ラベル背景
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                result, 
                (x, y - label_h - 10), 
                (x + label_w, y),
                color, -1
            )
            
            # ラベルテキスト
            cv2.putText(
                result, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return result
    
    def get_debug_masks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        デバッグ用：各色のマスクを取得
        
        Args:
            frame: BGR画像
            
        Returns:
            色名をキーとしたマスク画像の辞書
        """
        hsv = self._preprocess(frame)
        masks = {}
        
        for color_key in ['blue', 'yellow']:
            color_config = self.config[color_key]
            mask = self._create_color_mask(hsv, color_config)
            masks[color_key] = mask
        
        return masks
    
    def save_config(self, path: str):
        """設定をYAMLファイルに保存"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def update_hsv_range(
        self, 
        color: str, 
        lower: List[int], 
        upper: List[int]
    ):
        """HSV範囲を更新"""
        if color in self.config:
            self.config[color]['hsv_lower'] = lower
            self.config[color]['hsv_upper'] = upper


def create_default_config(output_path: str):
    """デフォルト設定ファイルを作成"""
    with open(output_path, 'w') as f:
        yaml.dump(ColorShapeDetector.DEFAULT_CONFIG, f, default_flow_style=False)
    print(f"Created default config: {output_path}")


if __name__ == "__main__":
    # デフォルト設定ファイル作成
    create_default_config("/workspace/scripts/detector_config.yaml")
