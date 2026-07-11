#!/usr/bin/env python3
"""
PP-OCRv6 モデルサイズ別ベンチマーク (medium / small / tiny)
複数レシートに対して 精度(フィールド抽出) / 推論時間 / メモリ(peak RSS) を計測する。

メモリを正しく分離計測するため、1プロセス=1モデルサイズで実行する:
    .venv/bin/python benchmark_models.py medium
    .venv/bin/python benchmark_models.py small
    .venv/bin/python benchmark_models.py tiny
"""
import sys
import time
import json
import resource
import cv2

from receipt_ocr_paddle import extract_fields, paddle_result_to_text

# 画像ごとの正解値(実写レシートの個人の購買データを含むため local_ground_truth.py
# (.gitignore対象)へ分離。無ければ実写評価は空dictとしてスキップされる)
try:
    from local_ground_truth import GROUND_TRUTHS
except ImportError:
    GROUND_TRUTHS = {}

PRESETS = {
    'medium': ('PP-OCRv6_medium_det', 'PP-OCRv6_medium_rec'),
    'small': ('PP-OCRv6_small_det', 'PP-OCRv6_small_rec'),
    'tiny': ('PP-OCRv6_tiny_det', 'PP-OCRv6_tiny_rec'),
}

def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else 'medium'
    det_model, rec_model = PRESETS[preset]

    from paddleocr import PaddleOCR
    t0 = time.time()
    ocr = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        # スマホ撮影は縦横回転が混在するため文書全体の向き分類を有効化
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    t_init = time.time() - t0

    report = {'preset': preset, 'det': det_model, 'rec': rec_model,
              'init_sec': round(t_init, 2), 'images': {}}

    for img_path, gt in GROUND_TRUTHS.items():
        img = cv2.imread(img_path)
        # ウォームアップ(初回はグラフ最適化等が乗るため計測から除外)
        ocr.predict(img)
        t = time.time()
        res = ocr.predict(img)
        infer_sec = time.time() - t

        text = paddle_result_to_text(res[0])
        fields = extract_fields(text)

        comparison = {}
        correct = 0
        for key, truth in gt.items():
            got = fields.get(key)
            ok = (got == truth)
            correct += ok
            comparison[key] = {'expected': truth, 'got': got, 'ok': ok}

        report['images'][img_path] = {
            'infer_sec': round(infer_sec, 2),
            'score': f"{correct}/{len(gt)}",
            'comparison': comparison,
            'raw_text': text,
        }

    report['peak_rss_mb'] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
