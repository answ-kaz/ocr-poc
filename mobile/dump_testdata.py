#!/usr/bin/env python3
"""
実レシートからモデル入力テンソルをダンプするスクリプト。

mobile/testdata/ に以下を保存:
  - receipt_receipt.jpg: receipt.jpg のテンソル群
  - receipt_receipt9.jpg: receipt9.jpg のテンソル群
  - 各テンソル: .npy + shapeメタ(.json)

使い方:
  .venv/bin/python mobile/dump_testdata.py
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from onnx_receipt_ocr import OnnxReceiptOCR, _imagenet_normalize
from ruri_experiment.onnx_line_labeler import OnnxLineLabeler, normalize_line

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_tensor(arr, name, meta):
    """テンソルを生バイナリ(.bin)で保存し、shape/dtypeメタも保存。

    npy形式はヘッダ(マジック+dict)が付きモバイル側のパースが煩雑なため、
    リトルエンディアンの生バイト列(tofile)+JSONメタの組で受け渡す。
    """
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(OUTPUT_DIR, f'{name}.bin'))
    meta['shape'] = list(arr.shape)
    meta['dtype'] = str(arr.dtype)
    with open(os.path.join(OUTPUT_DIR, f'{name}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  {name}: {arr.shape} {arr.dtype} ({arr.nbytes/1024:.1f}KB)')


def dump_receipt(img_path, prefix):
    """1枚のレシートから全テンソルをダンプ"""
    print(f'\n=== {img_path} → {prefix} ===')
    img = cv2.imread(img_path)
    ocr = OnnxReceiptOCR(quantized=True)
    labeler = OnnxLineLabeler()

    # 1. doc_ori入力
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    r = cv2.resize(img, (int(w * scale), int(h * scale)))
    rh, rw = r.shape[:2]
    top, left = (rh - 224) // 2, (rw - 224) // 2
    doc_input = _imagenet_normalize(r[top:top + 224, left:left + 224])
    save_tensor(doc_input, f'{prefix}_doc_ori', {'source': 'doc_ori_preprocess'})

    # 2. 文書向き補正
    oriented, doc_angle = ocr.correct_doc_orientation(img)

    # 3. det入力
    det_input = ocr._det_preprocess(oriented) if hasattr(ocr, '_det_preprocess') else None
    if det_input is None:
        # 手動でdet前処理
        h2, w2 = oriented.shape[:2]
        ratio = 1.0
        if max(h2, w2) > 960:
            ratio = 960 / max(h2, w2)
        rh2 = max(int(round(h2 * ratio / 32) * 32), 32)
        rw2 = max(int(round(w2 * ratio / 32) * 32), 32)
        resized = cv2.resize(oriented, (rw2, rh2))
        det_input = _imagenet_normalize(resized)
    save_tensor(det_input, f'{prefix}_det', {'doc_angle': doc_angle})

    # 4. 検出ボックス
    boxes = ocr.detect(oriented)

    # 5. rec入力 + textline_ori入力(cropごと)
    rec_inputs = []
    line_inputs = []
    for i, box in enumerate(boxes[:10]):  # 最大10crop
        crop = ocr._rotate_crop(oriented, box)
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        # rec入力
        h3, w3 = crop.shape[:2]
        img_w = int(np.clip(np.ceil(48 * w3 / h3), 16, 1920))
        r3 = cv2.resize(crop, (img_w, 48)).astype(np.float32) / 255.0
        rec_in = ((r3 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        rec_inputs.append(rec_in)

        # textline_ori入力
        line_crop = ocr._fix_line_orientation(crop)
        line_in = _imagenet_normalize(cv2.resize(line_crop, (160, 80)))
        line_inputs.append(line_in)

    if rec_inputs:
        # rec入力は幅が異なるため個別に保存
        for i, rec_in in enumerate(rec_inputs):
            save_tensor(rec_in, f'{prefix}_rec_{i}',
                         {'crop_index': i, 'n_crops': len(rec_inputs)})
    if line_inputs:
        save_tensor(np.concatenate(line_inputs, axis=0), f'{prefix}_textline',
                     {'n_crops': len(line_inputs)})

    # 6. ラベラー入力(実際のOCR行テキストから)。
    # 認識まで通してテキストを得る(rec_texts空のダミーでは行が出ない)
    from receipt_ocr_paddle import paddle_result_to_text
    res = ocr.predict(img)
    text = paddle_result_to_text(res)
    lines = [l for l in text.splitlines() if l.strip()][:20]  # 最大20行

    if lines:
        # トークン化・動的パディングはラベラー本体の実装をそのまま使う
        ids, mask = labeler._tokenize(lines)
        save_tensor(ids, f'{prefix}_labeler_ids', {'lines': lines})
        save_tensor(mask, f'{prefix}_labeler_mask', {})


def main():
    dump_receipt(os.path.join(ROOT, 'receipt.jpg'), 'receipt')
    dump_receipt(os.path.join(ROOT, 'receipt9.jpg'), 'receipt9')
    print(f'\n完了: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
