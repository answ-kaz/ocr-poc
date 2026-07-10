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

# 画像ごとの正解値(目視確認)。キーが無い項目はそのレシートに存在しない
GROUND_TRUTHS = {
    'receipt.jpg': {   # LAWSON ◯◯店
        'store_brand': 'LAWSON',
        'store_branch': '◯◯店',
        'registration_number': 'T0000000000001',
        'date': '2026-06-16',
        'time': '18:53',
        'total': 494,
        'tax': 36,
        'deposit': 10000,
        'change': 9506,
    },
    'receipt2.jpg': {  # セブン-イレブン ◯◯店(90度回転・PayPay支払で預り/釣り無し)
        'store_brand': 'セブン-イレブン',
        'store_branch': '◯◯店',
        'registration_number': 'T0000000000002',
        'date': '2026-04-16',
        'time': '10:29',
        'total': 289,
        'tax': 21,
    },
    'receipt3.jpg': {  # モスバーガー ◯◯店(ドライブスルー・クレジット払いで預り/釣り無し)
        'store_brand': 'MOS BURGER',
        'store_branch': '◯◯店',
        'registration_number': 'T0000000000003',
        'date': '2026-04-10',
        'time': '12:22',
        'total': 5240,
        'tax': 388,   # (内税) ¥388 = 軽減税8.0%
    },
    'receipt4.jpg': {  # LAWSON ◯◯店(地方公共団体証明書代の領収証・非課税・登録番号記載なし)
        'store_brand': 'LAWSON',
        'store_branch': '◯◯店',
        'date': '2026-07-10',
        'time': '18:30',
        'total': 200,
    },
    'receipt5.jpg': {  # LAWSON ◯◯店(メルペイ払いで預り/釣り無し)
        'store_brand': 'LAWSON',
        'store_branch': '◯◯店',
        'registration_number': 'T0000000000004',
        'date': '2026-07-07',
        'time': '09:19',
        'total': 1680,
        'tax': 144,   # 内消費税等
    },
    'receipt6.jpg': {  # うさちゃんクリーニング ◯◯店(クリーニング代領収書・時刻は「15時16分」表記)
        'store_brand': 'うさちゃんクリーニング',
        'store_branch': '◯◯店',
        'registration_number': 'T0000000000005',
        'date': '2026-06-16',
        'time': '15:16',
        'total': 2503,
        'tax': 228,   # 内税額10.0%
    },
}

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
