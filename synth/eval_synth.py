#!/usr/bin/env python3
"""
合成レシート評価 — synth/images/*.png を ONNXパイプラインでOCRし、
synth/gt.json と突き合わせて画像別スコアとフィールド別正解率を出力する。

実行: .venv/bin/python synth/eval_synth.py [--verbose]
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from onnx_receipt_ocr import OnnxReceiptOCR  # noqa: E402
from receipt_ocr_paddle import extract_fields, paddle_result_to_text  # noqa: E402

FIELD_ORDER = ['store_brand', 'store_branch', 'registration_number',
               'date', 'time', 'total', 'tax', 'deposit', 'change']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true', help='NG画像のOCR生テキストも表示')
    args = ap.parse_args()

    with open(os.path.join(ROOT, 'synth', 'gt.json'), encoding='utf-8') as f:
        gts = json.load(f)

    # 架空ブランドの店名マスタ(実運用のチェーン店マスタDB相当)を抽出側へ注入
    synth_brands = sorted({gt['store_brand'] for gt in gts.values() if 'store_brand' in gt})

    ocr = OnnxReceiptOCR(model_dir=os.path.join(ROOT, 'onnx_models'))

    field_stat = {k: [0, 0] for k in FIELD_ORDER}  # field -> [correct, present]
    total_correct = total_fields = 0
    ng_details = []

    print(f"{'画像':<16} {'スコア':>7}  NG項目")
    print('-' * 72)
    t0 = time.time()
    for name, gt in sorted(gts.items()):
        img = cv2.imread(os.path.join(ROOT, 'synth', 'images', name))
        res = ocr.predict(img)
        text = paddle_result_to_text(res)
        fields = extract_fields(text, extra_brands=synth_brands)

        ngs = []
        for k, truth in gt.items():
            got = fields.get(k)
            ok = (got == truth)
            field_stat[k][1] += 1
            field_stat[k][0] += ok
            if not ok:
                ngs.append((k, truth, got))
        total_correct += len(gt) - len(ngs)
        total_fields += len(gt)

        ng_str = ', '.join(f"{k}(exp={e} got={g})" for k, e, g in ngs)
        print(f"{name:<16} {len(gt)-len(ngs)}/{len(gt):<5}  {ng_str}")
        if ngs:
            ng_details.append((name, ngs, text))

    elapsed = time.time() - t0
    print('-' * 72)
    print(f"総合: {total_correct}/{total_fields} "
          f"({100*total_correct/total_fields:.1f}%)  {elapsed:.1f}s / {len(gts)}枚")

    print("\nフィールド別正解率:")
    print(f"{'field':<22} {'正解/出現':>10}  {'正解率':>7}")
    for k in FIELD_ORDER:
        c, n = field_stat[k]
        if n:
            print(f"{k:<22} {c:>5}/{n:<4}  {100*c/n:>6.1f}%")

    if args.verbose and ng_details:
        print("\n" + "=" * 72)
        print("NG画像のOCR生テキスト")
        for name, ngs, text in ng_details:
            print(f"\n--- {name} ---")
            for k, e, g in ngs:
                print(f"  NG {k}: expected={e} got={g}")
            print(text)


if __name__ == '__main__':
    main()
