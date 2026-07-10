#!/usr/bin/env python3
"""
ONNXモデルのINT8量子化 — スマホ同梱サイズ(FP32合計42.5MB→15MB前後)の検証用。

方式:
  - rec (PP-OCRv6_small_rec): dynamic量子化(MatMulのみ)と hybrid の両候補を生成。
    dynamicはMatMul重みのみINT8(conv部~10MBはFP32のまま、13.2MB)。
    hybrid = convのみstatic QDQ(Percentile 99.9) + MatMul dynamic の二段量子化(5.5MB)。
    全ノードstatic QDQ(MinMax)は「計→言」等の認識誤りが出たため不採用。
  - det (PP-OCRv6_small_det): conv主体でdynamic(MatMul)はほぼ無効果のため、
    実写レシート+合成画像をキャリブレーションデータにした static量子化(QDQ, per-channel)。
    比較用に dynamic版(*_dyn)も生成する。
  - 向き分類2種 (PP-LCNet): conv主体(HardSwish+SEで量子化に敏感)。static QDQは
    MinMaxだとFP32との判定一致率88%まで劣化するため Percentile(99.9) キャリブレーションを使用。
    キャリブレーションには正立に加え180度回転cropも含める(両クラスを網羅)。

det/doc_ori/line_ori/rec のキャリブレーション入力は onnx_receipt_ocr.py の
detect()/correct_doc_orientation()/_fix_line_orientation()/_recognize() と同一の前処理で作る。

生成物: onnx_models/*_int8.onnx (採用版) と onnx_models/*_int8_{dyn,qdq}.onnx (候補)
実行: .venv/bin/python quantize_models.py  (プロジェクトルートから)
"""
import glob
import os
import shutil

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from onnx_receipt_ocr import (DET_LIMIT_SIDE_LEN, MODEL_DIR, OnnxReceiptOCR,
                              _imagenet_normalize)

# キャリブレーションデータ: 実写レシート8枚 + 合成レシート20枚
CALIB_IMAGES = sorted(glob.glob('receipt*.jpg')) + sorted(glob.glob('synth/images/*.png'))

# 評価後に採用した方式(dyn / qdq / hybrid)。*_int8.onnx はこの選択のコピーとして生成される。
SELECTION = {
    'PP-OCRv6_small_rec': 'hybrid',
    'PP-OCRv6_small_det': 'qdq',
    'PP-LCNet_x1_0_doc_ori': 'qdq',
    'PP-LCNet_x1_0_textline_ori': 'qdq',
}


class ArrayReader(CalibrationDataReader):
    """前処理済みNCHW配列のリストをそのまま流すデータリーダー"""
    def __init__(self, arrays, input_name='x'):
        self._it = iter([{input_name: a} for a in arrays])

    def get_next(self):
        return next(self._it, None)


# ---------------- キャリブレーション入力の構築(FP32パイプラインの前処理を再現) ----------------
def det_preprocess(img):
    """onnx_receipt_ocr.OnnxReceiptOCR.detect() と同一の前処理"""
    h, w = img.shape[:2]
    ratio = 1.0
    if max(h, w) > DET_LIMIT_SIDE_LEN:
        ratio = DET_LIMIT_SIDE_LEN / max(h, w)
    rh = max(int(round(h * ratio / 32) * 32), 32)
    rw = max(int(round(w * ratio / 32) * 32), 32)
    resized = cv2.resize(img, (rw, rh))
    return _imagenet_normalize(resized)


def doc_ori_preprocess(img):
    """correct_doc_orientation() と同一の前処理(min辺256リサイズ+224センタークロップ)"""
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    r = cv2.resize(img, (int(w * scale), int(h * scale)))
    rh, rw = r.shape[:2]
    top, left = (rh - 224) // 2, (rw - 224) // 2
    return _imagenet_normalize(r[top:top + 224, left:left + 224])


def rec_preprocess(crop):
    """_recognize() 相当の前処理。ただしキャリブレーション用に幅を320へ固定
    (Percentileキャリブレーションはテンソル形状が全バッチ同一である必要があるため。
    3x48x320 はPaddleOCR recの学習時形状で、分布の近似として妥当)"""
    r = cv2.resize(crop, (320, 48)).astype(np.float32) / 255.0
    return ((r - 0.5) / 0.5).transpose(2, 0, 1)[None]


def build_calibration_inputs():
    """FP32パイプラインで文書向き補正・検出を行い、各モデルのキャリブレーション入力を作る"""
    fp32 = OnnxReceiptOCR()
    det_in, doc_in, line_in, rec_in = [], [], [], []
    for path in CALIB_IMAGES:
        img = cv2.imread(path)
        doc_in.append(doc_ori_preprocess(img))
        oriented, _ = fp32.correct_doc_orientation(img)
        det_in.append(det_preprocess(oriented))
        for box in fp32.detect(oriented):
            crop = fp32._rotate_crop(oriented, box)
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            # line_ori 入力: _fix_line_orientation() と同一のcrop。180度回転も加え両クラスを網羅
            for rot in (False, True):
                c = cv2.rotate(crop, cv2.ROTATE_180) if rot else crop
                line_in.append(_imagenet_normalize(cv2.resize(c, (160, 80))))
            # rec 入力: 行向き補正後のcrop(_recognize() の入力と同一)
            rec_in.append(rec_preprocess(fp32._fix_line_orientation(crop)))
    # line_oriはキャリブレーションのコストを抑えるためサンプリング(分布は十分カバーできる)。
    # recは精度マージンが小さいため全cropを使う(実写8+合成20で約840件)
    rng = np.random.default_rng(0)
    if len(line_in) > 400:
        line_in = [line_in[i] for i in rng.choice(len(line_in), 400, replace=False)]
    return det_in, doc_in, line_in, rec_in


# ---------------- 量子化 ----------------
def dynamic_quantize(src, dst):
    """MatMul重みのみINT8化(conv層はFP32のまま)。キャリブレーション不要"""
    quantize_dynamic(src, dst, op_types_to_quantize=['MatMul'],
                     per_channel=True, weight_type=QuantType.QInt8)


def static_quantize(src, dst, arrays, percentile=None, op_types=None):
    """QDQ形式のstatic量子化(activation/weightともS8、conv per-channel)。
    percentile指定時はMinMaxの代わりにPercentileキャリブレーションを使う
    (外れ値でレンジが引き伸ばされるのを防ぎ、量子化に敏感なモデルの精度が改善)。
    op_types指定時は対象オペレータを限定する(例: recはConvのみstatic化)"""
    import onnx
    from onnx import version_converter
    # per-channel QDQ (DequantizeLinearのaxis属性) は opset13以上が必要。
    # paddle2onnx出力は opset 9/11 のため先に変換する。
    up = dst + '.opset13.tmp.onnx'
    model = onnx.load(src)
    if model.opset_import[0].version < 13:
        model = version_converter.convert_version(model, 13)
    onnx.save(model, up)
    pre = dst + '.pre.tmp.onnx'
    try:
        # 形状推論+グラフ最適化の前処理(量子化精度・カバレッジが向上する公式推奨手順)
        quant_pre_process(up, pre, skip_symbolic_shape=True)
    except Exception as e:  # 前処理に失敗しても量子化自体は可能
        print(f"  quant_pre_process失敗({e}) — 元モデルを直接量子化")
        shutil.copyfile(up, pre)
    os.remove(up)
    kwargs = {}
    if percentile is not None:
        kwargs = dict(calibrate_method=CalibrationMethod.Percentile,
                      extra_options={'CalibPercentile': percentile})
    if op_types is not None:
        kwargs['op_types_to_quantize'] = op_types
    quantize_static(pre, dst, ArrayReader(arrays),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    **kwargs)
    os.remove(pre)


def mb(path):
    return os.path.getsize(path) / 1024 / 1024


def smoke_test(path, sample):
    """量子化モデルが読み込め、推論が通ることを確認"""
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    sess.run(None, {'x': sample})


def main():
    print(f"キャリブレーション画像: {len(CALIB_IMAGES)}枚")
    det_in, doc_in, line_in, rec_in = build_calibration_inputs()
    print(f"det入力 {len(det_in)}件 / doc_ori入力 {len(doc_in)}件 / "
          f"line_ori crop {len(line_in)}件 / rec crop {len(rec_in)}件")

    jobs = [
        # (モデル名, dynamic対象, staticキャリブ入力, percentile, static対象op)
        # rec: convのみstatic化(hybrid用)。全opのstatic化は認識誤りが出るため行わない
        ('PP-OCRv6_small_rec', True, rec_in, 99.9, ['Conv']),
        ('PP-OCRv6_small_det', True, det_in, None, None),  # MinMaxでFP32同等を確認済み
        ('PP-LCNet_x1_0_doc_ori', True, doc_in, 99.9, None),
        ('PP-LCNet_x1_0_textline_ori', True, line_in, 99.9, None),
    ]
    for name, do_dyn, calib, percentile, op_types in jobs:
        src = f'{MODEL_DIR}/{name}.onnx'
        variants = {}
        if do_dyn:
            dst = f'{MODEL_DIR}/{name}_int8_dyn.onnx'
            dynamic_quantize(src, dst)
            variants['dyn'] = dst
        if calib is not None and op_types is not None:
            # hybrid: convのみstatic QDQ → 残りのMatMulをdynamic化の二段量子化
            static_dst = f'{MODEL_DIR}/{name}_int8_convqdq.onnx'
            static_quantize(src, static_dst, calib, percentile=percentile, op_types=op_types)
            dst = f'{MODEL_DIR}/{name}_int8_hybrid.onnx'
            dynamic_quantize(static_dst, dst)
            os.remove(static_dst)
            variants['hybrid'] = dst
        elif calib is not None:
            dst = f'{MODEL_DIR}/{name}_int8_qdq.onnx'
            static_quantize(src, dst, calib, percentile=percentile)
            variants['qdq'] = dst

        sample = (calib[0] if calib is not None else
                  np.zeros((1, 3, 48, 320), dtype=np.float32))
        for kind, path in variants.items():
            smoke_test(path, sample)
            print(f"{name}: {mb(src):5.1f}MB → [{kind}] {mb(path):5.1f}MB")

        # 採用版を *_int8.onnx へコピー
        chosen = variants.get(SELECTION[name]) or next(iter(variants.values()))
        final = f'{MODEL_DIR}/{name}_int8.onnx'
        shutil.copyfile(chosen, final)
        print(f"  → 採用: {SELECTION[name]} を {os.path.basename(final)} にコピー")


if __name__ == '__main__':
    main()
