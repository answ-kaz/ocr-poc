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
        return {'rec_texts': texts, 'rec_polys': polys, 'doc_angle': doc_angle}


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
