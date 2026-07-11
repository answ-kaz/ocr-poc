#!/usr/bin/env python3
"""
レシートOCR ONNX版 — PaddlePaddle非依存、ONNX Runtime + OpenCV + pyclipperのみで動作。
スマホ組み込み(ONNX Runtime Mobile)と同じ構成のリファレンス実装兼フットプリント計測。

モデル: PP-OCRv6_small det/rec + PP-LCNet doc/textline orientation (paddle2onnxで変換済み)
パイプライン: 文書向き分類 → テキスト検出(DB) → 行向き分類 → 認識(CTC) → 行再構成 → フィールド抽出

実行: .venv/bin/python onnx_receipt_ocr.py
"""
import sys
import time
import json
import resource
import cv2
import numpy as np
import pyclipper
import onnxruntime as ort

from receipt_ocr_paddle import extract_fields, paddle_result_to_text
from benchmark_models import GROUND_TRUTHS

MODEL_DIR = 'onnx_models'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DBPostProcess パラメータ(PP-OCRv6_small_det/inference.yml より)
DET_THRESH = 0.2
DET_BOX_THRESH = 0.45
DET_UNCLIP_RATIO = 1.4
DET_MAX_CANDIDATES = 3000
# det入力は最大辺960に縮小(モバイル定番設定)。paddlexデフォルトの min辺736以上 だと
# ピークメモリが2倍(1.7GB)になるが精度は変わらなかったため縮小側を採用。
# 認識は原寸画像からの切り出しなので解像度低下の影響を受けない。
DET_LIMIT_SIDE_LEN = 960

# ---------------- 条件付き前処理(劣化画像ゲート) ----------------
# 常時前処理はDLモデルに有害(clean 30/33→29/33 を実測)のため、軽量な品質推定で
# 「劣化しているときだけ」適用する2段ゲート方式。閾値の根拠は external ベンチ
# (K10124 degraded 各10枚)+実写9枚+合成20枚での実測分布(下記)。
#
# ゲート1: ストロークコントラスト stroke_p95
#   最大辺 PRE_GATE_MAX_SIDE に縮小したグレースケールの 3x3 モルフォロジー勾配の
#   95パーセンタイル。掠れ・ブレ・低照度で低下する。実測分布:
#     clean 173-237 / 合成 201-255 / light 106-147 / medium 52-94 / heavy 39-76
#     実写 64-109
#   → 閾値 100 で clean / light / 合成 は全数ゲート外(前処理なし=回帰なし)。
# ゲート2: 最暗インク濃度 ink_p1
#   紙領域(局所背景輝度>=PRE_PAPER_LO)の flat=gray/bg の1パーセンタイル。
#   「印字自体が掠れて薄い」画像でのみ大きくなる。実測分布:
#     実写 0.05-0.30(印字は濃く背景が暗いだけ) / medium 0.30-0.54 / heavy 0.25-0.51
#   → 閾値 0.32 で実写9枚は全数ゲート外。medium/heavy は 9/10 枚が対象になる。
#   (ゲート1のみだと実写7枚が対象になり、INT8 で実写 57/57→55/57 に回帰した。
#    強調で太った文字を INT8 rec が誤読するため。ゲート2の追加で回帰ゼロ。)
PRE_GATE_MAX_SIDE = 640
PRE_GATE_STROKE_P95 = 100.0
PRE_GATE_INK_P1 = 0.32

# 前処理本体: モルフォロジー閉処理(最大辺256の縮小画像上、カーネル≒最大辺/PRE_BG_KERNEL_FRAC)
# で局所背景輝度(紙面+照明ムラ+帯影)を推定して除算 → ガンマ3.5で淡い印字を濃く戻す。
# 除算は紙領域(局所背景輝度が PRE_PAPER_LO..HI で高いところ)にのみ適用し、
# 暗い机・布背景はそのまま残す(全面除算だと背景テクスチャが増幅されゴミ検出が増え、
# 行再構成が壊れることを実測で確認: FP32 medium 18/33 → マスク付き 20/33)。
# ガンマは 2.0/3.0/3.5/4.0 を比較し 3.5 を採用
# (FP32 medium/heavy: 19/10, 19/13, 20/12, 20/10。INT8 では 3.5 が 19/11 で最良)。
# 常時二値化は有害と実証済みのため不使用。CLAHE(17/8)・アンシャープ併用(18/11-12)・
# 適応ガンマ(20/12だが実写INT8で52/57に悪化)・信頼度採択の二重推論(17/10。劣化画像でも
# 誤読を高信頼で返すため採択が機能しない)はいずれも比較の結果不採用。
PRE_BG_KERNEL_FRAC = 24     # 背景推定カーネル: 縮小画像最大辺の約1/24(1/12,1/40比較で最良)
PRE_ENHANCE_GAMMA = 3.5
PRE_PAPER_LO = 100.0        # 局所背景輝度がこの値以下 → 非紙領域として原画像を維持
PRE_PAPER_HI = 170.0        # この値以上 → 完全に除算+ガンマを適用(間は線形ブレンド)


def estimate_stroke_contrast(img):
    """品質ゲート1用の軽量指標: 文字ストロークのコントラスト(大きいほど鮮明)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = PRE_GATE_MAX_SIDE / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (max(int(w * scale), 8), max(int(h * scale), 8)))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return float(np.percentile(grad, 95))


def preprocess_if_degraded(img):
    """2段ゲートで劣化を判定し、該当時のみ 紙領域の背景除算+ガンマ強調 を適用する。

    返り値: (画像, 適用したか)。適用時はグレースケールの3ch複製を返す
    (レシートは実質無彩色で、det/rec/向き分類とも精度影響がないことを実測確認済み)。
    """
    if estimate_stroke_contrast(img) >= PRE_GATE_STROKE_P95:
        return img, False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 背景(紙面輝度)推定は最大辺256の縮小画像上で行い計算量を抑える(スマホ前提)
    scale = 256.0 / max(h, w)
    small = cv2.resize(gray, (max(int(w * scale), 8), max(int(h * scale), 8)))
    k = max(small.shape) // PRE_BG_KERNEL_FRAC * 2 + 1
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), k / 2.0)
    flat_small = np.clip(
        small.astype(np.float32) / np.maximum(bg_small.astype(np.float32), 16.0), 0.0, 1.0)
    paper_px = flat_small[bg_small.astype(np.float32) >= PRE_PAPER_LO]
    # ゲート2: 紙領域が見つからない、または最暗インクが既に濃い(掠れていない)なら適用しない
    if paper_px.size < 100 or float(np.percentile(paper_px, 1)) < PRE_GATE_INK_P1:
        return img, False
    bg = cv2.resize(bg_small, (w, h)).astype(np.float32)
    flat = np.clip(gray.astype(np.float32) / np.maximum(bg, 16.0), 0.0, 1.0)
    enhanced = flat ** PRE_ENHANCE_GAMMA * 255.0
    paper = np.clip((bg - PRE_PAPER_LO) / (PRE_PAPER_HI - PRE_PAPER_LO), 0.0, 1.0)
    out = paper * enhanced + (1.0 - paper) * gray.astype(np.float32)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), True


def _imagenet_normalize(img):
    """BGR uint8 HWC -> float32 NCHW (ImageNet mean/std)"""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None]


class OnnxReceiptOCR:
    def __init__(self, model_dir=MODEL_DIR, quantized=False):
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        providers = ['CPUExecutionProvider']
        # quantized=True で quantize_models.py が生成した INT8 モデル(*_int8.onnx)を読む
        suffix = '_int8' if quantized else ''
        self.det = ort.InferenceSession(f'{model_dir}/PP-OCRv6_small_det{suffix}.onnx', opt, providers=providers)
        self.rec = ort.InferenceSession(f'{model_dir}/PP-OCRv6_small_rec{suffix}.onnx', opt, providers=providers)
        self.doc_ori = ort.InferenceSession(f'{model_dir}/PP-LCNet_x1_0_doc_ori{suffix}.onnx', opt, providers=providers)
        self.line_ori = ort.InferenceSession(f'{model_dir}/PP-LCNet_x1_0_textline_ori{suffix}.onnx', opt, providers=providers)
        with open(f'{model_dir}/rec_dict.txt', encoding='utf-8') as f:
            chars = f.read().split('\n')
        # CTCLabelDecode: 先頭にblank、末尾に空白を追加
        self.characters = ['blank'] + chars + [' ']

    # ---------------- 文書向き分類 ----------------
    def correct_doc_orientation(self, img):
        h, w = img.shape[:2]
        scale = 256 / min(h, w)
        r = cv2.resize(img, (int(w * scale), int(h * scale)))
        rh, rw = r.shape[:2]
        top, left = (rh - 224) // 2, (rw - 224) // 2
        crop = r[top:top + 224, left:left + 224]
        logits = self.doc_ori.run(None, {'x': _imagenet_normalize(crop)})[0][0]
        angle = [0, 90, 180, 270][int(np.argmax(logits))]
        if angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img, angle

    # ---------------- テキスト検出 (DB) ----------------
    def detect(self, img):
        h, w = img.shape[:2]
        ratio = 1.0
        if max(h, w) > DET_LIMIT_SIDE_LEN:
            ratio = DET_LIMIT_SIDE_LEN / max(h, w)
        rh = max(int(round(h * ratio / 32) * 32), 32)
        rw = max(int(round(w * ratio / 32) * 32), 32)
        resized = cv2.resize(img, (rw, rh))
        prob = self.det.run(None, {'x': _imagenet_normalize(resized)})[0][0, 0]
        boxes = self._db_postprocess(prob)
        # 検出座標を元画像スケールへ戻す
        boxes[:, :, 0] = np.clip(boxes[:, :, 0] * w / rw, 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1] * h / rh, 0, h - 1)
        return boxes

    @staticmethod
    def _mini_box(contour):
        """minAreaRectの4点を [左上,右上,右下,左下] 順で返す"""
        rect = cv2.minAreaRect(contour)
        pts = sorted(cv2.boxPoints(rect), key=lambda p: p[0])
        i1, i4 = (0, 1) if pts[0][1] < pts[1][1] else (1, 0)
        i2, i3 = (2, 3) if pts[2][1] < pts[3][1] else (3, 2)
        box = np.array([pts[i1], pts[i2], pts[i3], pts[i4]], dtype=np.float32)
        return box, min(rect[1])

    def _db_postprocess(self, prob):
        bitmap = (prob > DET_THRESH).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[:DET_MAX_CANDIDATES]:
            box, sside = self._mini_box(contour)
            if sside < 3:
                continue
            # box_score_fast: 矩形内の確率平均
            xmin = int(np.clip(box[:, 0].min(), 0, prob.shape[1] - 1))
            xmax = int(np.clip(box[:, 0].max(), 0, prob.shape[1] - 1))
            ymin = int(np.clip(box[:, 1].min(), 0, prob.shape[0] - 1))
            ymax = int(np.clip(box[:, 1].max(), 0, prob.shape[0] - 1))
            mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            cv2.fillPoly(mask, [(box - [xmin, ymin]).astype(np.int32)], 1)
            score = cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
            if score < DET_BOX_THRESH:
                continue
            # unclip: 縮小されたテキスト領域を膨張して実サイズへ
            area = cv2.contourArea(box)
            length = cv2.arcLength(box, True)
            if length < 1e-6:
                continue
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(area * DET_UNCLIP_RATIO / length)
            if not expanded:
                continue
            box, sside = self._mini_box(np.array(expanded[0]).reshape(-1, 1, 2).astype(np.int32))
            if sside < 5:
                continue
            boxes.append(box)
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4, 2), dtype=np.float32)

    # ---------------- 切り出し・行向き・認識 ----------------
    @staticmethod
    def _rotate_crop(img, box):
        w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if h > 0 and w > 0 and h / w >= 1.5:
            crop = np.rot90(crop)
        return crop

    def _fix_line_orientation(self, crop):
        r = cv2.resize(crop, (160, 80))
        logits = self.line_ori.run(None, {'x': _imagenet_normalize(r)})[0][0]
        if int(np.argmax(logits)) == 1:  # '180_degree'
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        return crop

    def _recognize(self, crop):
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return '', 0.0
        img_w = int(np.clip(np.ceil(48 * w / h), 16, 1920))
        r = cv2.resize(crop, (img_w, 48)).astype(np.float32) / 255.0
        x = ((r - 0.5) / 0.5).transpose(2, 0, 1)[None]
        preds = self.rec.run(None, {'x': x})[0][0]  # (T, vocab)
        idx = preds.argmax(axis=-1)
        probs = preds.max(axis=-1)
        chars, scores = [], []
        prev = 0
        for i, p in zip(idx, probs):
            if i != 0 and i != prev:  # blank除去 + 重複除去
                chars.append(self.characters[i])
                scores.append(p)
            prev = i
        return ''.join(chars), float(np.mean(scores)) if scores else 0.0

    # ---------------- パイプライン ----------------
    def predict(self, img):
        # 条件付き前処理: 文書向き分類の「前」に適用する。劣化画像では向き分類自体が
        # 崩れる(180度/90度誤り→全行ゴミ化)ため、前段適用で heavy 11/33→13/33 と実測差が出た。
        img, preprocessed = preprocess_if_degraded(img)
        img, doc_angle = self.correct_doc_orientation(img)
        boxes = self.detect(img)
        texts, polys = [], []
        for box in boxes:
            crop = self._rotate_crop(img, box)
            crop = self._fix_line_orientation(crop)
            text, score = self._recognize(crop)
            if text:
                texts.append(text)
                polys.append(box)
        # receipt_ocr_paddle.paddle_result_to_text と互換の形式で返す
        return {'rec_texts': texts, 'rec_polys': polys, 'doc_angle': doc_angle,
                'preprocessed': preprocessed}


def main():
    ocr = OnnxReceiptOCR()

    print("=" * 60)
    print("ONNX Runtime版 (PP-OCRv6_small, PaddlePaddle非依存)")
    print(f"onnxruntime {ort.__version__}")
    print("=" * 60)

    for img_path, gt in GROUND_TRUTHS.items():
        img = cv2.imread(img_path)
        ocr.predict(img)  # ウォームアップ
        t = time.time()
        res = ocr.predict(img)
        infer_sec = time.time() - t

        text = paddle_result_to_text(res)
        fields = extract_fields(text)
        ngs = [(k, v, fields.get(k)) for k, v in gt.items() if fields.get(k) != v]

        print(f"\n--- {img_path} (文書向き: {res['doc_angle']}度, 推論 {infer_sec:.2f}s) ---")
        print(text)
        print(f"\nスコア: {len(gt) - len(ngs)}/{len(gt)}")
        for k, exp, got in ngs:
            print(f"  NG {k}: expected={exp} got={got}")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"\nピークメモリ(RSS): {peak_mb:.0f}MB")


if __name__ == '__main__':
    main()
