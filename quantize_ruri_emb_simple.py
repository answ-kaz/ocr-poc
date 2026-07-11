#!/usr/bin/env python3
"""
ruri-v3-30m 埋め込みテーブルのINT8化(per-rowスケール)。

model_int8.onnx(Dynamic量子化済み、FP32埋め込みテーブル100MBが残存)の
埋め込みを INT8テーブル+行スケール に置換し、Gatherの後段で
Cast→スケール乗算により FP32 へ復元するようグラフを配線し直す。
FP32テーブルを実行時に復元しない(gather後の該当トークン分のみ復元)ため、
モデルサイズと同時にピークRSSも下がる。

実行: .venv/bin/python quantize_ruri_emb_simple.py
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort

ONNX_DIR = 'onnx_models/ruri-v3-30m'
SRC_INT8 = os.path.join(ONNX_DIR, 'model_int8.onnx')
DST_INT8_EMB = os.path.join(ONNX_DIR, 'model_int8_emb8.onnx')
EMB_NAME = 'embeddings.tok_embeddings.weight'


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pool_embedding(output, attention_mask):
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


def main():
    model = onnx.load(SRC_INT8)

    # 埋め込みテーブルを取得
    emb_init = None
    for init in model.graph.initializer:
        if init.name == EMB_NAME:
            emb_init = init
            break

    if emb_init is None:
        print(f'{EMB_NAME} が見つかりません')
        return

    fp32_emb = numpy_helper.to_array(emb_init)
    print(f'元の埋め込み: shape={fp32_emb.shape}, dtype={fp32_emb.dtype}, '
          f'size={fp32_emb.nbytes/1024/1024:.1f}MB')

    # INT8量子化 (per-row)
    scales = np.abs(fp32_emb).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8)
    int8_emb = np.clip(np.round(fp32_emb / scales), -128, 127).astype(np.int8)

    print(f'INT8埋め込み: shape={int8_emb.shape}, dtype={int8_emb.dtype}, '
          f'size={int8_emb.nbytes/1024/1024:.1f}MB')
    print(f'スケール: shape={scales.shape}')

    # 復元精度テスト
    restored = int8_emb.astype(np.float32) * scales
    sims = []
    for i in range(min(100, len(fp32_emb))):
        sims.append(cosine_sim(fp32_emb[i], restored[i]))
    print(f'復元精度: mean={np.mean(sims):.6f} min={np.min(sims):.6f}')

    # グラフ配線: FP32テーブルを参照するGatherを
    #   Gather(int8_table) → Cast(float32) → Mul(Gather(row_scale)) に置換する
    consumers = [(i, n) for i, n in enumerate(model.graph.node)
                 if EMB_NAME in n.input]
    assert consumers, f'{EMB_NAME} を入力に持つノードが見つかりません'
    assert all(n.op_type == 'Gather' for _, n in consumers), \
        f'Gather以外の消費ノードあり: {[n.op_type for _, n in consumers]}'

    model.graph.initializer.remove(emb_init)
    model.graph.initializer.append(
        numpy_helper.from_array(int8_emb, name=EMB_NAME + '_int8'))
    model.graph.initializer.append(
        numpy_helper.from_array(scales.astype(np.float32), name=EMB_NAME + '_scale'))

    # インデックスが後ろへずれないよう降順で置換
    for idx, node in sorted(consumers, key=lambda t: -t[0]):
        ids_input = [x for x in node.input if x != EMB_NAME][0]
        out = node.output[0]
        axis = next((a.i for a in node.attribute if a.name == 'axis'), 0)
        new_nodes = [
            helper.make_node('Gather', [EMB_NAME + '_int8', ids_input],
                             [out + '_i8'], axis=axis),
            helper.make_node('Cast', [out + '_i8'], [out + '_f32'],
                             to=TensorProto.FLOAT),
            helper.make_node('Gather', [EMB_NAME + '_scale', ids_input],
                             [out + '_sc'], axis=axis),
            helper.make_node('Mul', [out + '_f32', out + '_sc'], [out]),
        ]
        model.graph.node.remove(node)
        for j, nn in enumerate(new_nodes):
            model.graph.node.insert(idx + j, nn)

    onnx.checker.check_model(model)
    onnx.save(model, DST_INT8_EMB)
    print(f'\n保存: {DST_INT8_EMB}')
    print(f'サイズ: {os.path.getsize(DST_INT8_EMB)/1024/1024:.1f}MB')

    # 実推論での等価性検証(サンプル文で元モデルとのcos類似度)
    samples = ['合計 ¥1,234', 'お預り ¥10,000', '2026年7月11日 18:36',
               'レギユラーガソリン P04 数量 ¥1000', '登録番号:T1234567890123']
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(ONNX_DIR, 'tokenizer.model'))
    s_src = ort.InferenceSession(SRC_INT8, providers=['CPUExecutionProvider'])
    s_dst = ort.InferenceSession(DST_INT8_EMB, providers=['CPUExecutionProvider'])
    sims = []
    for text in samples:
        ids = [sp.bos_id()] + sp.encode(text, out_type=int) + [sp.eos_id()]
        feeds = {'input_ids': np.array([ids], dtype=np.int64),
                 'attention_mask': np.ones((1, len(ids)), dtype=np.int64)}
        mask = np.ones(len(ids), dtype=np.int64)
        a = pool_embedding(s_src.run(None, feeds)[0][0], mask)
        b = pool_embedding(s_dst.run(None, feeds)[0][0], mask)
        sims.append(cosine_sim(a, b))
    print(f'実推論cos類似度(元model_int8比): mean={np.mean(sims):.6f} min={np.min(sims):.6f}')


if __name__ == '__main__':
    main()
