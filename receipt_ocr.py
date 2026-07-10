#!/usr/bin/env python3
"""
レシートOCR PoC
前処理: グレースケール化, 傾き補正, 二値化
OCR: Tesseract (jpn)
後処理: 店名/登録番号/日付/金額を正規表現で抽出
"""
import cv2
import numpy as np
import pytesseract
import re
import sys
import json

def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 適応的二値化(感熱紙のムラ・照明ムラに強い)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # 傾き補正(Hough変換で罫線/テキスト行の傾きを推定)
    coords = np.column_stack(np.where(binary < 128))
    angle = 0.0
    if len(coords) > 100:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        # 極端な誤検出を防ぐ(反りレシートは回転よりトリミング/歪み補正が本質のため角度は小さく制限)
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

    return rotated_gray, rotated_binary, angle

def ocr(image, psm=6):
    config = f"--oem 3 --psm {psm} -l jpn"
    text = pytesseract.image_to_string(image, config=config)
    return text

def extract_fields(text):
    result = {}

    # 店名(LAWSON等のブランド名、大文字英字の連続を抽出)
    m = re.search(r'[A-Z]{4,}', text)
    if m:
        result['store_brand'] = m.group()

    # 支店名(◯◯店で終わる行)
    m = re.search(r'\S+店', text)
    if m:
        result['store_branch'] = m.group()

    # 登録番号(インボイス制度 T+13桁)
    m = re.search(r'T\d{13}', text.replace(' ', ''))
    if m:
        result['registration_number'] = m.group()

    # 日付(YYYY年M月D日)
    m = re.search(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', text)
    if m:
        result['date'] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 時刻
    m = re.search(r'(\d{1,2}):(\d{2})', text)
    if m:
        result['time'] = m.group()

    # 金額系(合計・お預り・お釣り) - ¥または円 + 数字(カンマ区切り可)
    amounts = {}
    patterns = {
        'total': r'合\s計\s[¥￥]?\s*([\d,]+)',
        'deposit': r'お預[りか]\s*(?:合計)?\s*[¥￥]?\s*([\d,]+)',
        'change': r'お\s釣\s[り]?\s*[¥￥]?\s*([\d,]+)',
        'tax': r'消費税額?\s*[¥￥]?\s*([\d,]+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            amounts[key] = int(m.group(1).replace(',', ''))
    result['amounts'] = amounts

    return result

if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'receipt.jpg'

    gray, binary, angle = preprocess(img_path)
    cv2.imwrite('preprocessed_gray.png', gray)
    cv2.imwrite('preprocessed_binary.png', binary)

    print(f"=== 推定傾き角度: {angle:.2f}度 ===\n", file=sys.stderr)

    # 複数のPSM(ページ分割モード)と前処理パターンで試す
    candidates = []
    for label, image in [('gray', gray), ('binary', binary)]:
        for psm in [4, 6, 11]:
            text = ocr(image, psm=psm)
            candidates.append((f"{label}_psm{psm}", text))

    # 全結果を出力(比較用)
    for label, text in candidates:
        print(f"\n{'='*20} {label} {'='*20}")
        print(text)

    # 抽出は一番情報量が多そうな組み合わせ(binary+psm6)をベースに、
    # 足りない項目は他候補からフォールバック
    print(f"\n{'='*20} フィールド抽出結果 {'='*20}")
    merged_text = "\n".join(t for _, t in candidates)
    fields = extract_fields(merged_text)
    print(json.dumps(fields, ensure_ascii=False, indent=2))
