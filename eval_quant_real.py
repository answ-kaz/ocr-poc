#!/usr/bin/env python3
"""
FP32 vs INT8 の実写レシート評価ドライバー(量子化の精度切り分け用)。

構成を切り替えて GROUND_TRUTHS の全画像を評価し、スコア・推論時間・ピークRSSを出力する。
  fp32     : 全モデルFP32
  int8     : 全モデルINT8 (OnnxReceiptOCR(quantized=True))
  rec-int8 : recのみINT8(det/向き分類はFP32) — 劣化の切り分け用
  det-int8 : detのみINT8(rec/向き分類はFP32) — 劣化の切り分け用
さらに任意モデルの個別差し替え: --swap det=onnx_models/xxx.onnx 形式。

実行: .venv/bin/python eval_quant_real.py <config> [--swap name=path ...]
"""
import argparse
import json
import resource
import time

import cv2
import onnxruntime as ort

from benchmark_models import GROUND_TRUTHS
from onnx_receipt_ocr import MODEL_DIR, OnnxReceiptOCR
from receipt_ocr_paddle import extract_fields, paddle_result_to_text


def build_ocr(config, swaps):
    ocr = OnnxReceiptOCR(quantized=(config == 'int8'))
    def int8_session(name):
        return ort.InferenceSession(f'{MODEL_DIR}/{name}_int8.onnx',
                                    providers=['CPUExecutionProvider'])
    if config == 'rec-int8':
        ocr.rec = int8_session('PP-OCRv6_small_rec')
    elif config == 'det-int8':
        ocr.det = int8_session('PP-OCRv6_small_det')
    for name, path in swaps:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        setattr(ocr, name, sess)
    return ocr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', choices=['fp32', 'int8', 'rec-int8', 'det-int8'])
    ap.add_argument('--swap', action='append', default=[],
                    help='モデル個別差し替え: det=path.onnx / rec=... / doc_ori=... / line_ori=...')
    ap.add_argument('--json', action='store_true', help='集計をJSON1行で出力(比較表作成用)')
    args = ap.parse_args()
    swaps = [s.split('=', 1) for s in args.swap]

    ocr = build_ocr(args.config, swaps)

    total_ok = total_gt = 0
    times = []
    rows = []
    for img_path, gt in GROUND_TRUTHS.items():
        img = cv2.imread(img_path)
        ocr.predict(img)  # ウォームアップ
        t = time.time()
        res = ocr.predict(img)
        times.append(time.time() - t)
        fields = extract_fields(paddle_result_to_text(res))
        ngs = [(k, v, fields.get(k)) for k, v in gt.items() if fields.get(k) != v]
        total_ok += len(gt) - len(ngs)
        total_gt += len(gt)
        rows.append((img_path, len(gt) - len(ngs), len(gt), ngs))

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    if args.json:
        print(json.dumps({
            'config': args.config, 'swaps': swaps,
            'score': f'{total_ok}/{total_gt}',
            'per_image': {p: f'{ok}/{n}' for p, ok, n, _ in rows},
            'ng': {p: [f'{k}: {e} -> {g}' for k, e, g in ngs]
                   for p, _, _, ngs in rows if ngs},
            'avg_infer_sec': round(sum(times) / len(times), 3),
            'peak_rss_mb': round(peak_mb),
        }, ensure_ascii=False))
        return

    print(f"=== config: {args.config} {swaps if swaps else ''} ===")
    for img_path, ok, n, ngs in rows:
        print(f"{img_path}: {ok}/{n}")
        for k, e, g in ngs:
            print(f"  NG {k}: expected={e} got={g}")
    print(f"合計: {total_ok}/{total_gt}  平均推論 {sum(times)/len(times):.2f}s/枚  "
          f"ピークRSS {peak_mb:.0f}MB")


if __name__ == '__main__':
    main()
