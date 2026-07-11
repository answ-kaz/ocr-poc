#!/usr/bin/env python3
"""
レシートOCRパイプライン — camera_guide + regex + ruri fallback 完結版。

使い方:
  from pipeline import ReceiptOCR
  ocr = ReceiptOCR()
  result = ocr.process('receipt.jpg')
  if result.ok:
      print(result.fields)
  else:
      print(result.messages)  # ガイド表示

  # または画像配列から
  result = ocr.process_from_array(img_bgr)

実行:
  .venv/bin/python pipeline.py receipt.jpg  # テスト
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from camera_guide import ImageQualityGuide, format_guide
from onnx_receipt_ocr import OnnxReceiptOCR
from receipt_ocr_paddle import paddle_result_to_text, extract_fields
from ruri_experiment.onnx_line_labeler import (
    FALLBACK_FIELDS, apply_line_guard)


class OCRResult:
    """OCR結果"""
    def __init__(self):
        self.ok = False
        self.fields = {}
        self.text = ''
        self.messages = []
        self.metrics = {}
        self.source = ''  # 'regex' or 'fallback'

    def add_metric(self, key, value):
        self.metrics[key] = value


class ReceiptOCR:
    """レシートOCRパイプライン

    処理フロー:
      1. camera_guide で画像品質チェック
      2. OCR実行 (OnnxReceiptOCR)
      3. regex でフィールド抽出
      4. regex未検出フィールドを ruri ラベラーで補完
    """

    def __init__(self, use_camera_guide=True, threshold=0.80, quantized=False):
        self.use_camera_guide = use_camera_guide
        if use_camera_guide:
            self.guide = ImageQualityGuide()
        self.ocr = OnnxReceiptOCR(quantized=quantized)
        self._labeler = None
        self._labeler_threshold = threshold

    @property
    def labeler(self):
        """ラベラーは遅延初期化(使わない場合はロードしない)"""
        if self._labeler is None:
            from ruri_experiment.onnx_line_labeler import OnnxLineLabeler
            self._labeler = OnnxLineLabeler(threshold=self._labeler_threshold)
        return self._labeler

    def process(self, img_path):
        """画像パスから処理"""
        if self.use_camera_guide:
            qr = self.guide.check(img_path)
            if not qr.ok:
                result = OCRResult()
                result.messages = [format_guide(qr)]
                result.metrics = qr.metrics
                return result

        img = cv2.imread(img_path)
        if img is None:
            result = OCRResult()
            result.messages.append(f"画像を読み込めません: {img_path}")
            return result
        return self.process_from_array(img)

    def process_from_array(self, img_bgr):
        """BGR配列から処理"""
        result = OCRResult()
        t0 = time.time()

        # OCR実行
        res = self.ocr.predict(img_bgr)
        text = paddle_result_to_text(res)
        result.text = text
        result.add_metric('ocr_lines', len([l for l in text.splitlines() if l.strip()]))
        result.add_metric('ocr_sec', time.time() - t0)

        # regex抽出
        t1 = time.time()
        regex_fields = extract_fields(text)
        result.add_metric('regex_sec', time.time() - t1)

        # fallback抽出(ruriラベラー)。regex未検出フィールドがある場合のみ実行し、
        # ラベラーは遅延初期化した共有インスタンスを使う(毎回の110MBロードを防ぐ)
        t2 = time.time()
        merged = dict(regex_fields)
        added = []
        lines = [l for l in text.splitlines() if l.strip()]
        missing = [f for f in FALLBACK_FIELDS if f not in merged]
        if missing and lines:
            labels = apply_line_guard(lines, self.labeler.label_lines(lines))
            fallback_fields = self.labeler.extract_from_labels(lines, labels)
            for field in missing:
                if field in fallback_fields:
                    merged[field] = fallback_fields[field]
                    added.append(field)
        result.add_metric('labeler_sec', time.time() - t2)

        result.fields = merged
        result.ok = len(merged) > 0
        result.source = 'regex+fallback' if added else 'regex'
        result.add_metric('fields_count', len(merged))
        result.add_metric('fallback_added', len(added))
        result.add_metric('total_sec', time.time() - t0)

        return result


def main():
    if len(sys.argv) < 2:
        print("使い方: .venv/bin/python pipeline.py <画像パス>")
        sys.exit(1)

    ocr = ReceiptOCR()
    result = ocr.process(sys.argv[1])

    print("=== OCR結果 ===")
    if result.ok:
        print(f"抽出フィールド ({result.source}):")
        for k, v in result.fields.items():
            print(f"  {k}: {v}")
    else:
        print("不合格:")
        for m in result.messages:
            print(f"  {m}")

    print(f"\n=== メトリクス ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
