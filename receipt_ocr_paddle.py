#!/usr/bin/env python3
"""
レシートOCR PoC (PaddleOCR版 + Tesseract比較)
前処理: グレースケール化, 傾き補正, 二値化 (Tesseract版から流用)
OCR: PaddleOCR (lang=japan, デフォルトのPP-OCRv6_medium det/rec) を「前処理なし/gray/binary」の3パターンで実行
     Tesseract (jpn) を gray/binary × psm4/6/11 で実行(ベースライン比較用)
後処理: 店名/登録番号/日付/金額を正規表現で抽出し、正解値との比較表を出力
"""
import cv2
import numpy as np
import re
import sys
import json
import unicodedata

# ============================================================
# 正解値(receipt.jpg を目視確認した値)
# ============================================================
GROUND_TRUTH = {
    'store_brand': 'LAWSON',
    'store_branch': '◯◯店',
    'registration_number': 'T0000000000001',
    'date': '2026-06-16',
    'time': '18:53',
    'total': 494,
    'tax': 36,
    'deposit': 10000,
    'change': 9506,
}

FIELD_LABELS = [
    ('store_brand', '店名(ブランド)'),
    ('store_branch', '支店名'),
    ('registration_number', '登録番号'),
    ('date', '日付'),
    ('time', '時刻'),
    ('total', '合計'),
    ('tax', '消費税'),
    ('deposit', 'お預り'),
    ('change', 'お釣り'),
]

# ============================================================
# 前処理(Tesseract版 receipt_ocr.py から流用)
# ============================================================
def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 適応的二値化(感熱紙のムラ・照明ムラに強い)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # 傾き補正
    coords = np.column_stack(np.where(binary < 128))
    angle = 0.0
    if len(coords) > 100:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 5:
            angle = 0.0

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    rotated_binary = cv2.adaptiveThreshold(
        rotated_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    return img, rotated_gray, rotated_binary, angle

# ============================================================
# OCR実行
# ============================================================
def ocr_tesseract(image, psm=6):
    import pytesseract
    config = f"--oem 3 --psm {psm} -l jpn"
    return pytesseract.image_to_string(image, config=config)

def paddle_result_to_text(res):
    """PaddleOCRの検出ボックスをY座標でグルーピングして行テキストに再構成する。
    「合計」(左)と「¥494」(右)のように同一行が別ボックスになるため、
    正規表現マッチには行単位の結合が必須。"""
    texts = res['rec_texts']
    polys = res['rec_polys']
    if len(texts) == 0:
        return ""

    items = []
    heights = []
    for text, poly in zip(texts, polys):
        poly = np.asarray(poly)
        y_center = float(poly[:, 1].mean())
        x_left = float(poly[:, 0].min())
        h = float(poly[:, 1].max() - poly[:, 1].min())
        items.append((y_center, x_left, text))
        heights.append(h)

    line_tol = float(np.median(heights)) * 0.6
    items.sort(key=lambda t: t[0])

    lines = []
    current = [items[0]]
    for it in items[1:]:
        if it[0] - current[-1][0] <= line_tol:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)

    out = []
    for line in lines:
        line.sort(key=lambda t: t[1])
        out.append(" ".join(t[2] for t in line))
    return "\n".join(out)

def ocr_paddle(ocr_engine, image):
    """image: BGR 3ch ndarray"""
    results = ocr_engine.predict(image)
    return paddle_result_to_text(results[0])

# ============================================================
# フィールド抽出(全角/半角・字間スペース・カンマ区切りに対応)
# ============================================================
# 店名ブランド辞書(レシートは語彙が閉じているため辞書照合が確実)
BRAND_DICT = [
    'セブン-イレブン', 'ファミリーマート', 'FamilyMart', 'LAWSON', 'ローソン',
    'ミニストップ', 'デイリーヤマザキ', 'NewDays', 'セイコーマート',
]

def _compact(s):
    """ブランド照合用: 空白・ハイフン類・中点を除去"""
    return re.sub(r'[\s\-‐‑–—―・.]', '', s)

_PA_TO_BA = str.maketrans('パピプペポ', 'バビブベボ')

def _loose(s):
    """ブランド照合用の緩い正規化: OCRが混同しやすい半濁点→濁点を同一視"""
    return _compact(s).translate(_PA_TO_BA)

def extract_fields(text):
    # 全角英数字・全角記号(¥含む)を半角へ正規化
    text = unicodedata.normalize('NFKC', text)
    # OCR典型誤認識の正規化(「計」が偏旁分割で「言十」等になるケース)
    text = text.replace('言十', '計').replace('百十', '計')
    result = {}

    # 店名ブランド(辞書照合を優先、フォールバックは大文字英字の連続)
    text_loose = _loose(text)
    brand = None
    for b in BRAND_DICT:
        if _loose(b) in text_loose:
            brand = b
            break
    if brand is None:
        # 完全一致しない場合は類似度でファジーマッチ(1文字の脱落・置換を救う)
        from difflib import SequenceMatcher
        best = 0.0
        for b in BRAND_DICT:
            bl = _loose(b)
            for i in range(max(1, len(text_loose) - len(bl) + 1)):
                ratio = SequenceMatcher(None, bl, text_loose[i:i + len(bl)]).ratio()
                if ratio > best and ratio >= 0.8:
                    best, brand = ratio, b
    if brand:
        result['store_brand'] = brand
    else:
        m = re.search(r'[A-Z]{4,}', text)
        if m:
            result['store_brand'] = m.group()

    # 支店名(「◯◯店」で終わる語。ブランド名を除去してから照合)
    for line in text.split('\n'):
        compact = re.sub(r'\s+', '', line)
        if brand:
            compact = compact.replace(_compact(brand), '').replace(brand, '')
        m = re.search(r'([ぁ-んァ-ヶ一-龥A-Za-z0-9ー]+店)$', compact)
        if m and '対象' not in compact:
            result['store_branch'] = m.group(1)
            break

    # 登録番号(インボイス T+13桁。スペース除去後にマッチ)
    m = re.search(r'T\d{13}', re.sub(r'\s', '', text))
    if m:
        result['registration_number'] = m.group()

    # 日付(YYYY年M月D日)
    m = re.search(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', text)
    if m:
        result['date'] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 時刻(電話番号等と混同しないよう「:」区切りのみ)
    m = re.search(r'(\d{1,2})\s*[::]\s*(\d{2})(?!\d)', text)
    if m:
        result['time'] = f"{int(m.group(1)):02d}:{m.group(2)}"

    # 金額系。「合 計」「お 釣」の字間スペース、カンマ区切り、¥/￥有無に対応
    # 桁区切りはカンマの他、OCRで「.」に化けるケースも許容(円に小数は無いため安全)
    NUM = r'[¥\\]?\s*(\d{1,3}(?:[,，.]\d{3})+|\d+)'
    patterns = {
        # 「お預り合計」の合計にマッチしないよう直前の「り」を除外
        'total': r'(?<!り)合\s*計\s*' + NUM,
        # 「消費税等(8%) ¥21」のような税率括弧表記を許容
        'tax': r'内?\s*消\s*費\s*税\s*[等額]?\s*(?:\(?\s*\d{1,2}\s*%\s*\)?)?\s*' + NUM,
        'deposit': r'お\s*預\s*[りか]?\s*(?:合\s*計)?\s*' + NUM,
        'change': r'お\s*[釣鈎釘勺]\s*り?\s*' + NUM,
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[key] = int(re.sub(r'[,，.]', '', m.group(1)))

    # 整合性補正: 合計行が検出漏れしても「合計 = お預り − お釣り」で導出可能
    if 'total' not in result and 'deposit' in result and 'change' in result:
        derived = result['deposit'] - result['change']
        if derived > 0:
            result['total'] = derived

    return result

# ============================================================
# 比較表
# ============================================================
def build_comparison_table(results_by_engine):
    """results_by_engine: {エンジン名: 抽出結果dict}"""
    engines = list(results_by_engine.keys())
    header = "| 項目 | 正解値 | " + " | ".join(engines) + " |"
    sep = "|---" * (len(engines) + 2) + "|"
    rows = [header, sep]
    score = {e: 0 for e in engines}
    for key, label in FIELD_LABELS:
        truth = GROUND_TRUTH[key]
        cells = []
        for e in engines:
            got = results_by_engine[e].get(key)
            if got is None:
                cells.append("❌ (未検出)")
            elif got == truth:
                cells.append(f"✅ {got}")
                score[e] += 1
            else:
                cells.append(f"❌ {got}")
        rows.append(f"| {label} | {truth} | " + " | ".join(cells) + " |")
    total = len(FIELD_LABELS)
    rows.append("| **正解数** | - | " + " | ".join(f"**{score[e]}/{total}**" for e in engines) + " |")
    return "\n".join(rows)

# ============================================================
# メイン
# ============================================================
def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'receipt.jpg'

    print("=" * 60)
    print("前処理(グレースケール化・適応的二値化・傾き補正)")
    print("=" * 60)
    raw_bgr, gray, binary, angle = preprocess(img_path)
    cv2.imwrite('preprocessed_gray.png', gray)
    cv2.imwrite('preprocessed_binary.png', binary)
    print(f"推定傾き角度: {angle:.2f}度")

    # ---- PaddleOCR ----
    print("\n" + "=" * 60)
    print("PaddleOCR (lang=japan, デフォルト検出/認識モデル=PP-OCRv6_medium) 初期化")
    print("=" * 60)
    import paddle
    from paddleocr import PaddleOCR
    print(f"paddlepaddle {paddle.__version__} / device: {paddle.device.get_device()}")
    ocr_engine = PaddleOCR(
        lang='japan',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    paddle_inputs = [
        ('Paddle(前処理なし)', raw_bgr),
        ('Paddle(gray)', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ('Paddle(binary)', cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
    ]
    paddle_texts = {}
    for label, image in paddle_inputs:
        text = ocr_paddle(ocr_engine, image)
        paddle_texts[label] = text
        print(f"\n{'=' * 20} {label} 認識テキスト {'=' * 20}")
        print(text)

    # ---- Tesseract(ベースライン)----
    print("\n" + "=" * 60)
    print("Tesseract (jpn) ベースライン実行 (gray/binary × psm4/6/11)")
    print("=" * 60)
    tess_candidates = []
    for label, image in [('gray', gray), ('binary', binary)]:
        for psm in [4, 6, 11]:
            text = ocr_tesseract(image, psm=psm)
            tess_candidates.append((f"{label}_psm{psm}", text))
    for label, text in tess_candidates:
        print(f"\n{'=' * 20} Tesseract {label} {'=' * 20}")
        print(text)
    # ベースライン同様、全候補をマージして抽出(Tesseractに最も有利な条件)
    tess_merged = "\n".join(t for _, t in tess_candidates)

    # ---- 抽出・比較 ----
    print("\n" + "=" * 60)
    print("フィールド抽出結果 (JSON)")
    print("=" * 60)
    results = {'Tesseract(全candidateマージ)': extract_fields(tess_merged)}
    for label, text in paddle_texts.items():
        results[label] = extract_fields(text)
    for engine, fields in results.items():
        print(f"\n--- {engine} ---")
        print(json.dumps(fields, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("比較表 (正解値 vs 各エンジン)")
    print("=" * 60)
    print(build_comparison_table(results))

if __name__ == '__main__':
    main()
