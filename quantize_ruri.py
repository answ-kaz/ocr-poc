#!/usr/bin/env python3
"""
ruri-v3-30m ONNXモデルのINT8量子化。

方式:
  - dynamic量子化: キャリブレーション不要。MatMul重みをINT8化。
  - static量子化(QDQ): キャリブレーションデータでactivationレンジを決定。
    レシートOCR行テキストをプロトタイプ文+実際のOCR行でキャリブレーション。

生成物:
  - onnx_models/ruri-v3-30m/model_int8.onnx (dynamic)
  - onnx_models/ruri-v3-30m/model_int8_static.onnx (static QDQ)

実行: .venv/bin/python quantize_ruri.py
"""
import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod,
    QuantFormat, QuantType, quantize_dynamic, quantize_static
)

ONNX_DIR = 'onnx_models/ruri-v3-30m'
SRC_MODEL = os.path.join(ONNX_DIR, 'model.onnx')
DST_DYNAMIC = os.path.join(ONNX_DIR, 'model_int8.onnx')
DST_STATIC = os.path.join(ONNX_DIR, 'model_int8_static.onnx')

# キャリブレーションテキスト: プロトタイプ文 + 実際のOCR行サンプル
CALIB_TEXTS = [
    # プロトタイプ文(ラベラーが扱う行種)
    '合計 ¥494', '合 計 ¥1,800', '合計金額 ¥2,503', 'お買上げ計 ¥1,234',
    'お預り ¥1,000', 'お預かり ¥10,000', '現金 ¥6,000',
    'お釣り ¥506', 'おつり ¥0', 'お釣 ¥9,506',
    '(内消費税等 ¥36)', '消費税等(8%) ¥21', '内税額 10.0% ¥228',
    '2026年 6月16日(火) 18:53', '2026/07/10 20:59', '26.07.07 09:19',
    '新宿東口店', '渋谷駅前店', '本町一丁目店',
    '登録番号 T1234567890123', '事業者登録番号 T9876543210987',
    # ディストラクタ
    '電話：018-853-0502', 'TEL 0120-134-890',
    '東京都新宿区西新宿1-2-3 ビル2F', '小計 ¥268',
    '10%対象 ¥3,150', 'コーヒー ¥130', '牛乳 1点 ¥208',
    'PayPay支払 ¥289', 'クレジット支払 ¥5,240', 'ポイント残高 120P',
    '点数 2個', '上記正に領収いたしました',
    # 実際のOCR行パターン(架空店名で再現)
    'サンクスマート 桜ヶ丘一丁目店', 'T1234567890123',
    '2026年06月16日(火) 18:53', '合 計 ¥494', 'お預り ¥10,000', 'お釣 ¥9,506',
    'コーヒー 牛乳 ¥494', '(内消費税等 ¥36)',
    'クイックマート 緑町病院前店', 'T2345678901234',
    '2026年04月10日 12:22', '合計 ¥1,080', '現金 ¥2,000', 'お釣り ¥920',
]


class TextCalibrationReader(CalibrationDataReader):
    """テキストをトークン化してONNX入力を作成するキャリブレーションリーダー"""
    def __init__(self, tokenizer, texts, max_length=128):
        self._inputs = []
        for text in texts:
            enc = tokenizer(text, max_length=max_length, padding='max_length',
                            truncation=True, return_tensors='np')
            self._inputs.append({
                'input_ids': enc['input_ids'].astype(np.int64),
                'attention_mask': enc['attention_mask'].astype(np.int64),
            })
        self._it = iter(self._inputs)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._inputs)


def dynamic_quantize():
    """dynamic量子化 (キャリブレーション不要)"""
    quantize_dynamic(
        SRC_MODEL, DST_DYNAMIC,
        op_types_to_quantize=['MatMul', 'Gemm'],
        per_channel=True, weight_type=QuantType.QInt8
    )


def static_quantize():
    """static QDQ量子化"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    reader = TextCalibrationReader(tokenizer, CALIB_TEXTS)

    quantize_static(
        SRC_MODEL, DST_STATIC, reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={'CalibPercentile': 99.9}
    )


def pool_embedding(output, attention_mask):
    """last token pooling + attention_maskでパディング除去 → L2正規化。
    SentenceTransformerのPooling(mean)と同等の結果にはならないが、
    量子化精度の比較には十分。"""
    # output: (1, seq_len, dim) or (seq_len, dim)
    if output.ndim == 3:
        output = output[0]  # (seq_len, dim)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[0]  # (seq_len,)
    mask = attention_mask.astype(bool)
    if mask.any():
        emb = output[mask].mean(axis=0)
    else:
        emb = output.mean(axis=0)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def verify_quantized(path, tokenizer):
    """量子化モデルの推論確認 + プーリング済み埋め込みを返す"""
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    test_text = '合計 ¥1,800'
    enc = tokenizer(test_text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='np')
    feeds = {
        'input_ids': enc['input_ids'].astype(np.int64),
        'attention_mask': enc['attention_mask'].astype(np.int64),
    }
    out = sess.run(None, feeds)
    emb = pool_embedding(out[0], enc['attention_mask'])
    print(f"  出力shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
    return emb


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)

    # FP32基準
    print("=== FP32 ベースライン ===")
    fp32_emb = verify_quantized(SRC_MODEL, tokenizer)

    # dynamic量子化
    print("\n=== Dynamic INT8 量子化 ===")
    dynamic_quantize()
    dyn_emb = verify_quantized(DST_DYNAMIC, tokenizer)
    dyn_sim = cosine_sim(fp32_emb, dyn_emb)
    dyn_size = os.path.getsize(DST_DYNAMIC) / 1024 / 1024
    print(f"  FP32とのcos類似度: {dyn_sim:.6f}")
    print(f"  サイズ: {dyn_size:.1f}MB")

    # static量子化
    print("\n=== Static QDQ INT8 量子化 ===")
    static_quantize()
    stat_emb = verify_quantized(DST_STATIC, tokenizer)
    stat_sim = cosine_sim(fp32_emb, stat_emb)
    stat_size = os.path.getsize(DST_STATIC) / 1024 / 1024
    print(f"  FP32とのcos類似度: {stat_sim:.6f}")
    print(f"  サイズ: {stat_size:.1f}MB")

    # 類似度テスト
    print("\n=== 類似度テスト ===")
    texts = [
        '合計 ¥1,800',
        'お預り ¥2,000',
        '2026年06月16日 18:53',
        'コーヒー ¥130',
    ]
    encs = [tokenizer(t, max_length=128, padding='max_length',
                       truncation=True, return_tensors='np') for t in texts]

    def get_emb(sess, enc):
        feeds = {
            'input_ids': enc['input_ids'].astype(np.int64),
            'attention_mask': enc['attention_mask'].astype(np.int64),
        }
        out = sess.run(None, feeds)
        return pool_embedding(out[0], enc['attention_mask'])

    fp32_sess = ort.InferenceSession(SRC_MODEL, providers=['CPUExecutionProvider'])
    dyn_sess = ort.InferenceSession(DST_DYNAMIC, providers=['CPUExecutionProvider'])
    stat_sess = ort.InferenceSession(DST_STATIC, providers=['CPUExecutionProvider'])

    fp32_embs = [get_emb(fp32_sess, e) for e in encs]
    dyn_embs = [get_emb(dyn_sess, e) for e in encs]
    stat_embs = [get_emb(stat_sess, e) for e in encs]

    for i, t in enumerate(texts):
        for j in range(i + 1, len(texts)):
            fp32_sim = cosine_sim(fp32_embs[i], fp32_embs[j])
            dyn_sim_ij = cosine_sim(dyn_embs[i], dyn_embs[j])
            stat_sim_ij = cosine_sim(stat_embs[i], stat_embs[j])
            print(f"  [{i}]'{t[:15]}' vs [{j}]'{texts[j][:15]}': "
                  f"FP32={fp32_sim:.4f} dyn={dyn_sim_ij:.4f} stat={stat_sim_ij:.4f}")


if __name__ == '__main__':
    main()
