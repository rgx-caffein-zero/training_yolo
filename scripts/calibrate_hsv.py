#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSV Color Calibration Tool

トラックバーを使ってHSV色範囲を調整するツール
実際のシールの色に合わせてパラメータを調整するために使用
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path


class HSVCalibrator:
    """HSV色範囲キャリブレーションツール"""
    
    def __init__(self, source):
        """
        Args:
            source: 画像パスまたは動画パス、またはカメラID（int）
        """
        self.source = source
        self.is_camera = isinstance(source, int)
        self.is_video = False
        
        if not self.is_camera:
            path = Path(source)
            video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            self.is_video = path.suffix.lower() in video_ext
        
        # HSV範囲の初期値
        self.hsv_params = {
            'h_min': 0, 'h_max': 179,
            's_min': 0, 's_max': 255,
            'v_min': 0, 'v_max': 255
        }
        
        # ウィンドウ名
        self.window_main = 'Original'
        self.window_mask = 'Mask'
        self.window_result = 'Result'
        self.window_trackbar = 'HSV Controls'
    
    def _create_trackbars(self):
        """トラックバーを作成"""
        cv2.namedWindow(self.window_trackbar)
        
        cv2.createTrackbar('H min', self.window_trackbar, 0, 179, self._on_trackbar)
        cv2.createTrackbar('H max', self.window_trackbar, 179, 179, self._on_trackbar)
        cv2.createTrackbar('S min', self.window_trackbar, 0, 255, self._on_trackbar)
        cv2.createTrackbar('S max', self.window_trackbar, 255, 255, self._on_trackbar)
        cv2.createTrackbar('V min', self.window_trackbar, 0, 255, self._on_trackbar)
        cv2.createTrackbar('V max', self.window_trackbar, 255, 255, self._on_trackbar)
    
    def _on_trackbar(self, val):
        """トラックバーコールバック（何もしない）"""
        pass
    
    def _get_trackbar_values(self):
        """トラックバーの値を取得"""
        self.hsv_params['h_min'] = cv2.getTrackbarPos('H min', self.window_trackbar)
        self.hsv_params['h_max'] = cv2.getTrackbarPos('H max', self.window_trackbar)
        self.hsv_params['s_min'] = cv2.getTrackbarPos('S min', self.window_trackbar)
        self.hsv_params['s_max'] = cv2.getTrackbarPos('S max', self.window_trackbar)
        self.hsv_params['v_min'] = cv2.getTrackbarPos('V min', self.window_trackbar)
        self.hsv_params['v_max'] = cv2.getTrackbarPos('V max', self.window_trackbar)
    
    def _set_trackbar_values(self, h_min, h_max, s_min, s_max, v_min, v_max):
        """トラックバーの値を設定"""
        cv2.setTrackbarPos('H min', self.window_trackbar, h_min)
        cv2.setTrackbarPos('H max', self.window_trackbar, h_max)
        cv2.setTrackbarPos('S min', self.window_trackbar, s_min)
        cv2.setTrackbarPos('S max', self.window_trackbar, s_max)
        cv2.setTrackbarPos('V min', self.window_trackbar, v_min)
        cv2.setTrackbarPos('V max', self.window_trackbar, v_max)
    
    def _process_frame(self, frame):
        """フレームを処理"""
        # ブラー
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # HSV変換
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # マスク作成
        lower = np.array([
            self.hsv_params['h_min'],
            self.hsv_params['s_min'],
            self.hsv_params['v_min']
        ])
        upper = np.array([
            self.hsv_params['h_max'],
            self.hsv_params['s_max'],
            self.hsv_params['v_max']
        ])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # モルフォロジー変換
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 結果
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 輪郭描画と頂点数表示
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面積フィルタ
                # 輪郭近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                # バウンディングボックス
                x, y, w, h = cv2.boundingRect(contour)
                
                # 描画
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    result, f"v={vertices} a={int(area)}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
        
        return mask, result
    
    def _print_current_values(self):
        """現在の値を表示"""
        print("\n" + "=" * 50)
        print("Current HSV Range:")
        print("=" * 50)
        print(f"H: [{self.hsv_params['h_min']}, {self.hsv_params['h_max']}]")
        print(f"S: [{self.hsv_params['s_min']}, {self.hsv_params['s_max']}]")
        print(f"V: [{self.hsv_params['v_min']}, {self.hsv_params['v_max']}]")
        print("=" * 50)
        print("\nYAML format:")
        print(f"hsv_lower: [{self.hsv_params['h_min']}, {self.hsv_params['s_min']}, {self.hsv_params['v_min']}]")
        print(f"hsv_upper: [{self.hsv_params['h_max']}, {self.hsv_params['s_max']}, {self.hsv_params['v_max']}]")
    
    def _save_config(self, output_path: str, color_name: str):
        """設定をファイルに保存"""
        config = {
            color_name: {
                'hsv_lower': [
                    self.hsv_params['h_min'],
                    self.hsv_params['s_min'],
                    self.hsv_params['v_min']
                ],
                'hsv_upper': [
                    self.hsv_params['h_max'],
                    self.hsv_params['s_max'],
                    self.hsv_params['v_max']
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Saved to: {output_path}")
    
    def run(self):
        """キャリブレーションを実行"""
        print("=" * 50)
        print("HSV Color Calibration Tool")
        print("=" * 50)
        print("Controls:")
        print("  q     - Quit")
        print("  p     - Print current values")
        print("  s     - Save config to file")
        print("  b     - Set Blue preset")
        print("  y     - Set Yellow preset")
        print("  r     - Reset to full range")
        print("=" * 50)
        
        # ソースを開く
        if self.is_camera or self.is_video:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"Cannot open: {self.source}")
                return
        else:
            frame = cv2.imread(self.source)
            if frame is None:
                print(f"Cannot read image: {self.source}")
                return
        
        self._create_trackbars()
        
        while True:
            # フレーム取得
            if self.is_camera or self.is_video:
                ret, frame = cap.read()
                if not ret:
                    if self.is_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ループ
                        continue
                    break
            
            # トラックバーの値を取得
            self._get_trackbar_values()
            
            # 処理
            mask, result = self._process_frame(frame)
            
            # 情報表示
            info_text = f"H:[{self.hsv_params['h_min']}-{self.hsv_params['h_max']}] "
            info_text += f"S:[{self.hsv_params['s_min']}-{self.hsv_params['s_max']}] "
            info_text += f"V:[{self.hsv_params['v_min']}-{self.hsv_params['v_max']}]"
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 表示
            cv2.imshow(self.window_main, frame)
            cv2.imshow(self.window_mask, mask)
            cv2.imshow(self.window_result, result)
            
            # キー入力
            key = cv2.waitKey(1 if (self.is_camera or self.is_video) else 30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                self._print_current_values()
            elif key == ord('s'):
                color_name = input("Enter color name (e.g., blue, yellow): ").strip()
                output_path = input("Enter output file path: ").strip()
                if color_name and output_path:
                    self._save_config(output_path, color_name)
            elif key == ord('b'):
                # Blue preset
                self._set_trackbar_values(100, 130, 100, 255, 100, 255)
                print("Set Blue preset")
            elif key == ord('y'):
                # Yellow preset
                self._set_trackbar_values(20, 35, 100, 255, 100, 255)
                print("Set Yellow preset")
            elif key == ord('r'):
                # Reset
                self._set_trackbar_values(0, 179, 0, 255, 0, 255)
                print("Reset to full range")
        
        # クリーンアップ
        if self.is_camera or self.is_video:
            cap.release()
        cv2.destroyAllWindows()
        
        # 最終値を表示
        self._print_current_values()


def main():
    parser = argparse.ArgumentParser(description="HSV Color Calibration Tool")
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="0",
        help="Input source: image path, video path, or camera ID (default: 0)"
    )
    
    args = parser.parse_args()
    
    # カメラIDかパスかを判定
    try:
        source = int(args.input)
    except ValueError:
        source = args.input
    
    calibrator = HSVCalibrator(source)
    calibrator.run()


if __name__ == "__main__":
    main()
