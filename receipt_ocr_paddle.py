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

# 同一行とみなす垂直オーバーラップ率(小さい方のボックス高さに対する重なりの割合)
LINE_OVERLAP_RATIO = 0.5

def paddle_result_to_text(res):
    """PaddleOCRの検出ボックスをY座標でグルーピングして行テキストに再構成する。
    「合計」(左)と「¥494」(右)のように同一行が別ボックスになるため、
    正規表現マッチには行単位の結合が必須。
    同一行判定は「ボックスペアの垂直オーバーラップ率(小さい方の高さ基準)が閾値以上」
    かつ「互いのY中心が相手のY範囲内にある(相互中心包含)」で行う。
    中央値高さ×係数のY中心距離+チェーン結合だと、大フォント行(例:「合 計」)が
    前後の小フォント行と融合してしまうため、フォントサイズ差に頑健なこの方式を採用。
    相互中心包含は、検出ボックスのunclip膨張で密な隣接行同士が5割超重なるケース
    (行間の詰まった住所ブロック等)を分離するためのガード。"""
    texts = res['rec_texts']
    polys = res['rec_polys']
    if len(texts) == 0:
        return ""

    items = []
    for text, poly in zip(texts, polys):
        poly = np.asarray(poly)
        y_top = float(poly[:, 1].min())
        y_bottom = float(poly[:, 1].max())
        x_left = float(poly[:, 0].min())
        items.append((y_top, y_bottom, x_left, text))

    items.sort(key=lambda t: (t[0] + t[1]) / 2)

    # 各行は {'y0','y1'}=メンバーの平均区間を保持。候補ボックスとの重なりを
    # min(候補高さ, 行高さ) で割った比率が閾値以上なら同一行(最良の行へ割当)。
    # 平均区間で比較することで、行区間が union で肥大して別行を巻き込むのを防ぐ。
    lines = []
    for y0, y1, x_left, text in items:
        center = (y0 + y1) / 2
        best_line, best_ratio = None, LINE_OVERLAP_RATIO
        for line in lines:
            overlap = min(y1, line['y1']) - max(y0, line['y0'])
            denom = min(y1 - y0, line['y1'] - line['y0'])
            if denom <= 0:
                continue
            ratio = overlap / denom
            line_center = (line['y0'] + line['y1']) / 2
            mutual = (line['y0'] <= center <= line['y1']) and (y0 <= line_center <= y1)
            if mutual and ratio >= best_ratio:
                best_line, best_ratio = line, ratio
        if best_line is None:
            lines.append({'y0': y0, 'y1': y1, 'members': [(x_left, text)]})
        else:
            n = len(best_line['members'])
            best_line['y0'] = (best_line['y0'] * n + y0) / (n + 1)
            best_line['y1'] = (best_line['y1'] * n + y1) / (n + 1)
            best_line['members'].append((x_left, text))

    lines.sort(key=lambda l: (l['y0'] + l['y1']) / 2)
    out = []
    for line in lines:
        members = sorted(line['members'], key=lambda t: t[0])
        out.append(" ".join(t[1] for t in members))
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
    'MOS BURGER', 'モスバーガー', 'うさちゃんクリーニング',
    'Ringer Hut', 'リンガーハット', 'LUPICIA', 'ルピシア',
]

def _compact(s):
    """ブランド照合用: 空白・ハイフン類・長音・中点を除去
    (長音「ー」はOCRで半角「-」に誤認されやすいためハイフン類と同一視)"""
    return re.sub(r'[\s\-‐‑–—―ー・.]', '', s)

_PA_TO_BA = str.maketrans('パピプペポ', 'バビブベボ')
_BA_TO_PA = str.maketrans('バビブベボ', 'パピプペポ')

def _loose(s):
    """ブランド照合用の緩い正規化: OCRが混同しやすい半濁点→濁点を同一視"""
    return _compact(s).translate(_PA_TO_BA)

def _strip_brand(compact_line, brand):
    """行文字列(空白除去済み)からブランド名を除去する。
    「ー」と「-」の混同、長音の脱落・挿入、空白挿入、半濁点/濁点の誤認に頑健。"""
    sep = r'[\s\-‐‑–—―ー・.]*'
    parts = []
    for c in _compact(brand):
        variants = {c, c.translate(_PA_TO_BA), c.translate(_BA_TO_PA)}
        parts.append('[' + ''.join(re.escape(v) for v in sorted(variants)) + ']'
                     if len(variants) > 1 else re.escape(c))
    out = re.sub(sep.join(parts), '', compact_line, count=1)
    if out != compact_line:
        return out
    # 正規表現で除去できない崩れ方(「クリーニング→クニング」等の文字脱落)は
    # 類似度のスライディング窓で最良一致箇所を探して除去する
    from difflib import SequenceMatcher
    bl = _loose(brand)
    if len(bl) < 4:
        return compact_line
    best = (0.0, None, None)
    for size in range(max(3, len(bl) - 2), len(bl) + 3):
        for i in range(len(compact_line) - size + 1):
            ratio = SequenceMatcher(None, bl, _loose(compact_line[i:i + size])).ratio()
            if ratio > best[0]:
                best = (ratio, i, size)
    if best[0] >= 0.8:
        i, size = best[1], best[2]
        return compact_line[:i] + compact_line[i + size:]
    return compact_line

# OCRが簡体字グリフとして誤認識しやすい字の対応表(NFKCでは正規化されないため明示置換)
_SIMPLIFIED_TO_JP = str.maketrans('额领对费减轻', '額領対費減軽')

def extract_fields(text, extra_brands=None):
    # 全角英数字・全角記号(¥含む)を半角へ正規化
    text = unicodedata.normalize('NFKC', text)
    # OCR典型誤認識の正規化(「計」が偏旁分割で「言十」等になるケース)
    text = text.replace('言十', '計').replace('百十', '計')
    # 簡体字グリフへの誤認識を日本語字体へ戻す(例: 额→額)
    text = text.translate(_SIMPLIFIED_TO_JP)
    result = {}

    # 店名ブランド(辞書照合を優先、フォールバックは大文字英字の連続)
    # extra_brands: 店名マスタの外部注入(評価用データの架空店名等。実運用ではDB相当)
    brands = list(BRAND_DICT) + list(extra_brands or [])
    text_loose = _loose(text)
    brand = None
    for b in brands:
        if _loose(b) in text_loose:
            brand = b
            break
    if brand is None:
        # 完全一致しない場合は類似度でファジーマッチ(1文字の脱落・置換を救う)
        from difflib import SequenceMatcher
        best = 0.0
        for b in brands:
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
    # 先頭は数字以外に限定(行融合等で隣接した番地の数字を巻き込まないようガード)
    for line in text.split('\n'):
        compact = re.sub(r'\s+', '', line)
        if brand:
            compact = _strip_brand(compact, brand)
        # 半角「-」も許容(長音「ー」の誤認識対策)し、マッチ後にカタカナ間の「-」を長音へ戻す
        m = re.search(r'([ぁ-んァ-ヶ一-龥A-Za-z][ぁ-んァ-ヶ一-龥A-Za-z0-9ー\-]*店)$', compact)
        if m and '対象' not in compact:
            result['store_branch'] = re.sub(r'(?<=[ァ-ヶ])-(?=[ァ-ヶ])', 'ー', m.group(1))
            break

    # 登録番号(インボイス T+13桁。スペース除去後にマッチ)
    m = re.search(r'T\d{13}', re.sub(r'\s', '', text))
    if m:
        result['registration_number'] = m.group()

    # 日付(YYYY年M月D日 / YYYY/M/D / YYYY-M-D / YYYY.M.D / YY.M.D)
    date_patterns = [
        r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日',
        # 区切り記号形式。電話番号・管理番号等の数字列を拾わないよう前後の数字を禁止するが、
        # 「2026/06/1511:14」のように時刻が密着して認識されるケースは許容する
        r'(?<!\d)(\d{4})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?=\D|$|\d{1,2}\s*[::])',
        # 2桁年はドット区切りのみ許容(ハイフンは住所・電話番号と衝突するため)
        r'(?<![\d.])(\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})(?=\D|$|\d{1,2}\s*[::])',
    ]
    for pat in date_patterns:
        m = re.search(pat, text)
        if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
            year = int(m.group(1))
            if year < 100:
                year += 2000
            result['date'] = f"{year}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            break

    # 時刻(電話番号等と混同しないよう「:」区切りのみ。「HH時MM分」表記も対応)
    m = re.search(r'(\d{1,2})\s*[::]\s*(\d{2})(?!\d)', text) \
        or re.search(r'(\d{1,2})\s*時\s*(\d{1,2})\s*分', text)
    if m:
        result['time'] = f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

    # 金額系。「合 計」「お 釣」の字間スペース、カンマ区切り、¥/￥有無に対応
    # 桁区切りはカンマの他、OCRで「.」に化けるケースも許容(円に小数は無いため安全)。
    # 「¥3, 150」のように区切り直後にスペースが入る誤認識も許容
    NUM = r'[¥\\]?\s*(\d{1,3}(?:[,，.]\s*\d{3})+|\d+)'
    patterns = {
        # 「お預り合計」の合計にマッチしないよう直前の「り」を除外
        # 「合計金額 ¥2,503」のように「計」と金額の間に「金額」が入る表記と、
        # 「<合 計> ¥1,800」のような括弧付きラベルの閉じ記号を許容
        'total': r'(?<!り)合\s*計\s*[>)\]]?\s*(?:金\s*額)?\s*' + NUM,
        # 「お預り」「お預かり」「お預り合計」と、現金払いの「現金 ¥6,000」行に対応
        'deposit': r'(?:お\s*預\s*か?\s*り?\s*(?:合\s*計)?|現\s*金)\s*' + NUM,
        # ひらがな「おつり ¥0」表記も対応
        'change': r'お\s*[釣鈎釘勺つ]\s*り?\s*' + NUM,
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[key] = int(re.sub(r'[,，.\s]', '', m.group(1)))

    # 税は全マッチを収集(外税で税率別の行しか無いレイアウトに対応)。
    # 「消費税等(8%) ¥21」の税率括弧表記、「(内税) ¥388」「内税額10.0%¥228」、
    # 小数点付き税率を許容。税率なしの表記(=合算値であることが多い)を優先し、
    # 無ければ税率ごとに1件ずつ採って合算する(8%/10%併記の軽減税率レシート)。
    # 税率はラベル後の「消費税等(8%)」に加え、ラベル前の「10%対象内消費税」形式も拾う
    tax_pat = (r'(?:(?P<rate_pre>\d{1,2}(?:\.\d+)?)\s*%\s*対象\s*)?'
               r'(?:内?\s*消\s*費\s*税\s*[等額]?|内\s*税\s*額?)\s*\)?\s*'
               r'(?:\(?\s*(?P<rate>\d{1,2}(?:\.\d+)?)\s*%\s*\)?)?\s*'
               + NUM.replace('(', '(?P<amt>', 1))
    no_rate_amt, by_rate = None, {}
    for m in re.finditer(tax_pat, text):
        amt = int(re.sub(r'[,，.\s]', '', m.group('amt')))
        rate = m.group('rate') or m.group('rate_pre')
        if rate is None:
            if no_rate_amt is None:
                no_rate_amt = amt
        else:
            by_rate.setdefault(float(rate), amt)
    if no_rate_amt is not None:
        result['tax'] = no_rate_amt
    elif by_rate:
        result['tax'] = sum(by_rate.values())

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
