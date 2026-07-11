#!/usr/bin/env python3
"""
カメラガイドモジュール — 撮影時の画像品質チェック。

チェック項目:
  1. 明るさ (平均輝度)
  2. 傾き (Hough変換による罫線検出)
  3. 解像度 (短辺ピクセル数)
  4. ブレ (ラプラシアン分散)
  5. 文字領域検出 (EASTテキスト検出器)

使い方:
  from camera_guide import check_image_quality, ImageQualityGuide
  guide = ImageQualityGuide()
  result = guide.check(img_path)
  if not result.ok:
      print(result.messages)

実行:
  .venv/bin/python camera_guide.py receipt.jpg  # テスト
"""
import os
import sys

import cv2
import numpy as np


class QualityResult:
    """品質チェック結果"""
    def __init__(self):
        self.ok = True
        self.messages = []
        self.metrics = {}

    def fail(self, msg):
        self.ok = False
        self.messages.append(msg)

    def warn(self, msg):
        self.messages.append(f"[警告] {msg}")

    def add_metric(self, key, value):
        self.metrics[key] = value


class ImageQualityGuide:
    """画像品質チェック + ガイドメッセージ生成"""

    def __init__(self,
                 min_brightness=60,      # 平均輝度の下限
                 max_brightness=220,     # 平均輝度の上限(白飛び防止)
                 max_tilt_deg=15,        # 許容傾き(度)
                 min_short_edge=400,      # 短辺のハード下限(これ未満は拒否)
                 warn_short_edge=640,     # 短辺の警告閾値
                 min_sharpness=50,        # ラプラシアン分散の警告閾値
                 east_path=None):         # EASTモデルパス
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_tilt_deg = max_tilt_deg
        self.min_short_edge = min_short_edge
        self.warn_short_edge = warn_short_edge
        self.min_sharpness = min_sharpness
        self.east = None
        if east_path and os.path.exists(east_path):
            self.east = cv2.dnn.readNet(east_path)

    def check(self, img_path_or_array):
        """画像をチェックしQualityResultを返す"""
        result = QualityResult()

        if isinstance(img_path_or_array, str):
            img = cv2.imread(img_path_or_array)
            if img is None:
                result.fail(f"画像を読み込めません: {img_path_or_array}")
                return result
        else:
            img = img_path_or_array

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. 解像度チェック
        # ハード拒否は400px未満のみ。520-544pxの外部ベンチ画像はOCRが正常処理できる
        # ことを実測済みのため、640px未満は警告に留める
        short_edge = min(h, w)
        result.add_metric('short_edge', short_edge)
        if short_edge < self.min_short_edge:
            result.fail(f"解像度不足: 短辺{short_edge}px < {self.min_short_edge}px")
        elif short_edge < self.warn_short_edge:
            result.warn(f"解像度低め: 短辺{short_edge}px")

        # 2. 明るさチェック
        brightness = float(np.mean(gray))
        result.add_metric('brightness', brightness)
        if brightness < self.min_brightness:
            result.fail(f"暗すぎます: 平均輝度{brightness:.0f} < {self.min_brightness}")
        elif brightness > self.max_brightness:
            result.warn(f"明るすぎます(白飛びの可能性): 平均輝度{brightness:.0f}")

        # 3. ブレ検出(ラプラシアン分散) — 警告のみ、ハード拒否しない。
        # 実測でこの指標はOCR可否を分離できない: OCR満点の実写が21.4/46.3を示す一方、
        # OCRが崩れるheavy劣化画像はJPEGノイズで135-170に上振れし閾値50を通過する。
        # ライブビューでの手ブレ喚起としては有用なので警告表示のみ残す
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        result.add_metric('sharpness', laplacian_var)
        if laplacian_var < self.min_sharpness:
            result.warn(f"ブレの可能性: シャープネス{laplacian_var:.1f}")

        # 4. 傾き検出
        tilt_deg = self._detect_tilt(gray)
        result.add_metric('tilt_deg', tilt_deg)
        if abs(tilt_deg) > self.max_tilt_deg:
            result.fail(f"傾きすぎます: {tilt_deg:.1f}度 > ±{self.max_tilt_deg}度")

        # 5. テキスト領域検出(EASTがある場合のみ)
        if self.east is not None:
            text_ratio = self._detect_text_ratio(img)
            result.add_metric('text_ratio', text_ratio)
            if text_ratio < 0.01:
                result.warn("テキスト領域が検出できません")

        # ガイドメッセージ生成
        if not result.ok:
            result.messages.insert(0, "=== 撮影ガイド ===")
            if any("暗すぎ" in m for m in result.messages):
                result.messages.append("→ 明るい場所で撮り直してください")
            if any("明るすぎ" in m for m in result.messages):
                result.messages.append("→ 直射日光を避けて撮り直してください")
            if any("解像度不足" in m for m in result.messages):
                result.messages.append("→ カメラを近づけて撮り直してください")
            if any("ブレの可能性" in m for m in result.messages):
                result.messages.append("→ カメラを固定して撮り直してください")
            if any("傾きすぎます" in m for m in result.messages):
                result.messages.append("→ レシートをまっすぐに揃えて撮り直してください")

        return result

    def _detect_tilt(self, gray):
        """Hough変換でテキスト行の傾きを検出(度)"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 50:  # ほぼ水平な線のみ
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:  # 極端な斜めは除外
                    angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))

    def _detect_text_ratio(self, img):
        """EASTでテキスト領域の割合を検出"""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320),
                                     (123.68, 116.78, 103.94),
                                     True, False)
        self.east.setInput(blob)
        output = self.east.forward(['feature_fusion/Conv_7/Sigmoid',
                                     'feature_fusion/concat_3'])
        scores = output[0][0, 0, :, :]
        text_mask = (scores > 0.5).astype(np.uint8)
        return float(np.mean(text_mask))


def format_guide(result):
    """QualityResultを見やすい文字列にフォーマット"""
    lines = []
    lines.append(f"品質チェック: {'✓ 合格' if result.ok else '✗ 不合格'}")
    for k, v in result.metrics.items():
        if k == 'brightness':
            lines.append(f"  明るさ: {v:.0f}/255")
        elif k == 'sharpness':
            lines.append(f"  シャープネス: {v:.1f}")
        elif k == 'tilt_deg':
            lines.append(f"  傾き: {v:.1f}度")
        elif k == 'short_edge':
            lines.append(f"  解像度: {v}px")
        elif k == 'text_ratio':
            lines.append(f"  テキスト領域: {v:.1%}")
    for msg in result.messages:
        lines.append(f"  {msg}")
    return "\n".join(lines)


# ---------------- テスト ----------------
def main():
    if len(sys.argv) < 2:
        print("使い方: .venv/bin/python camera_guide.py <画像パス>")
        sys.exit(1)

    guide = ImageQualityGuide()
    result = guide.check(sys.argv[1])
    print(format_guide(result))

    if not result.ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
