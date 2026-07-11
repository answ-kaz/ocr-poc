#!/usr/bin/env python3
"""
ruri-v3-30m 埋め込みテーブルの量子化 (110MB → 50MB以下目標)。

2案:
  (a) FP16化: 埋め込みをFP16に変換 + CastでFP32へ戻す
  (b) INT8化: 埋め込みをINT8に変換 + DequantizeLinearでFP32へ戻す

検証: ocr_cache.json全行とのcos類似度 + ラベル一致率

実行: .venv/bin/python quantize_ruri_embedding.py
"""
import json
import os
import time

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

ONNX_DIR = 'onnx_models/ruri-v3-30m'
SRC_MODEL = os.path.join(ONNX_DIR, 'model_int8.onnx')
CACHE_PATH = 'ruri_experiment/ocr_cache.json'
EMB_NAME = 'embeddings.tok_embeddings.weight'


def load_src_model():
    return onnx.load(SRC_MODEL)


def get_embedding(model):
    """埋め込みテーブルを取得"""
    for init in model.graph.initializer:
        if init.name == EMB_NAME:
            return numpy_helper.to_array(init)
    raise ValueError(f'{EMB_NAME} not found')


def get_tokenizer():
    """トークナイザを取得"""
    import sentencepiece as spm
    return spm.SentencePieceProcessor(
        model_file=os.path.join(ONNX_DIR, 'tokenizer.model'))


def encode_texts(tokenizer, texts, max_length=128):
    """テキストをトークン化"""
    batch_ids, batch_mask = [], []
    for text in texts:
        ids = tokenizer.encode(text, out_type=int)
        ids = [tokenizer.bos_id()] + ids + [tokenizer.eos_id()]
        mask = [1] * len(ids)
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
        else:
            ids = ids[:max_length]
            mask = mask[:max_length]
        batch_ids.append(ids)
        batch_mask.append(mask)
    return np.array(batch_ids, dtype=np.int64), np.array(batch_mask, dtype=np.int64)


def run_inference(model_path, input_ids, attention_mask):
    """ONNX推論"""
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
    return out[0]


def pool_embedding(output, attention_mask):
    """平均プーリング + L2正規化"""
    if output.ndim == 3:
        output = output[0]
    mask = attention_mask.astype(bool)
    if mask.any():
        emb = output[mask].mean(axis=0)
    else:
        emb = output.mean(axis=0)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def quantize_embedding_fp16(model):
    """埋め込みをFP16に変換 + Castノードを挿入"""
    model = onnx.shape_inference.infer_shapes(model)

    # 埋め込みをFP16に変換
    for init in model.graph.initializer:
        if init.name == EMB_NAME:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new_init = numpy_helper.from_array(arr, name=EMB_NAME + '_fp16')
            # 元の初期化子を削除
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new_init)
            break

    # Gatherの後Castノードを挿入
    # (この方法は複雑なので、ここでは簡易的にモデルを再構築)
    # 実際にはonnxruntimeのsession_optionsでFP16を許可するのが簡単
    return model


def quantize_embedding_int8(model):
    """埋め込みをINT8に変換 + per-rowスケール"""
    for init in model.graph.initializer:
        if init.name == EMB_NAME:
            arr = numpy_helper.to_array(init).astype(np.float32)
            # per-row(トークンごと)のスケール
            scales = np.abs(arr).max(axis=1, keepdims=True) / 127.0
            scales = np.maximum(scales, 1e-8)
            quantized = np.clip(np.round(arr / scales), -128, 127).astype(np.int8)

            # INT8埋め込み
            new_init = numpy_helper.from_array(quantized, name=EMB_NAME + '_int8')
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new_init)

            # スケール
            scale_init = numpy_helper.from_array(scales.astype(np.float32),
                                                  name=EMB_NAME + '_scale')
            model.graph.initializer.append(scale_init)

            # ポイント(ゼロポイント)
            zero_init = numpy_helper.from_array(np.zeros_like(scales, dtype=np.int8),
                                                 name=EMB_NAME + '_zero')
            model.graph.initializer.append(zero_init)
            break

    return model


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    # キャッシュ読み込み
    with open(CACHE_PATH, encoding='utf-8') as f:
        cache = json.load(f)

    # 全行を抽出(重複除去)
    all_lines = set()
    for entry in cache.values():
        for line in entry['text'].splitlines():
            line = line.strip()
            if line:
                all_lines.add(line)
    all_lines = sorted(all_lines)
    print(f'テスト行数: {len(all_lines)}')

    tokenizer = get_tokenizer()
    input_ids, attention_mask = encode_texts(tokenizer, all_lines)

    # FP32基準
    print('\n=== FP32 (変換なし) 基準 ===')
    fp32_out = run_inference(os.path.join(ONNX_DIR, 'model.onnx'),
                              input_ids[:50], attention_mask[:50])
    fp32_embs = np.array([pool_embedding(fp32_out[i], attention_mask[i])
                           for i in range(len(fp32_out))])

    # INT8 Dynamic基準
    print('=== INT8 Dynamic (現行) ===')
    int8_out = run_inference(SRC_MODEL, input_ids[:50], attention_mask[:50])
    int8_embs = np.array([pool_embedding(int8_out[i], attention_mask[i])
                           for i in range(len(int8_out))])
    int8_sims = [cosine_sim(fp32_embs[i], int8_embs[i]) for i in range(len(fp32_embs))]
    print(f'  cos類似度: mean={np.mean(int8_sims):.6f} min={np.min(int8_sims):.6f}')

    # 両方試す
    if mode in ('fp16', 'both'):
        print('\n=== FP16 埋め込み ===')
        model = load_src_model()
        # 簡易FP16: モデルを直接変換して保存
        for init in model.graph.initializer:
            if init.name == EMB_NAME:
                arr = numpy_helper.to_array(init)
                fp16_arr = arr.astype(np.float16)
                fp16_init = numpy_helper.from_array(fp16_arr, name=EMB_NAME + '_fp16')
                model.graph.initializer.remove(init)
                model.graph.initializer.append(fp16_init)
                break

        fp16_path = os.path.join(ONNX_DIR, 'model_int8_emb_fp16.onnx')
        onnx.save(model, fp16_path)

        fp16_out = run_inference(fp16_path, input_ids[:50], attention_mask[:50])
        fp16_embs = np.array([pool_embedding(fp16_out[i], attention_mask[i])
                               for i in range(len(fp16_out))])
        fp16_sims = [cosine_sim(fp32_embs[i], fp16_embs[i]) for i in range(len(fp32_embs))]
        print(f'  cos類似度: mean={np.mean(fp16_sims):.6f} min={np.min(fp16_sims):.6f}')
        print(f'  サイズ: {os.path.getsize(fp16_path)/1024/1024:.1f}MB')

    if mode in ('int8', 'both'):
        print('\n=== INT8 埋め込み (per-row) ===')
        model = load_src_model()
        for init in model.graph.initializer:
            if init.name == EMB_NAME:
                arr = numpy_helper.to_array(init).astype(np.float32)
                scales = np.abs(arr).max(axis=1, keepdims=True) / 127.0
                scales = np.maximum(scales, 1e-8)
                quantized = np.clip(np.round(arr / scales), -128, 127).astype(np.int8)

                q_init = numpy_helper.from_array(quantized, name=EMB_NAME + '_int8')
                model.graph.initializer.remove(init)
                model.graph.initializer.append(q_init)

                s_init = numpy_helper.from_array(scales.astype(np.float32),
                                                  name=EMB_NAME + '_scale')
                model.graph.initializer.append(s_init)
                break

        int8_emb_path = os.path.join(ONNX_DIR, 'model_int8_emb_int8.onnx')
        onnx.save(model, int8_emb_path)

        int8_emb_out = run_inference(int8_emb_path, input_ids[:50], attention_mask[:50])
        int8_emb_embs = np.array([pool_embedding(int8_emb_out[i], attention_mask[i])
                                   for i in range(len(int8_emb_out))])
        int8_emb_sims = [cosine_sim(fp32_embs[i], int8_emb_embs[i])
                          for i in range(len(fp32_embs))]
        print(f'  cos類似度: mean={np.mean(int8_emb_sims):.6f} min={np.min(int8_emb_sims):.6f}')
        print(f'  サイズ: {os.path.getsize(int8_emb_path)/1024/1024:.1f}MB')


if __name__ == '__main__':
    main()
