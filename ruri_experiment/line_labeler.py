#!/usr/bin/env python3
"""
ruri-v3-310m によるレシートOCR行ラベリング実験。

方式:
  - 各フィールドの「プロトタイプ文」(実際のレシート行を模した例文)を埋め込み、
    OCR行との cos類似度の最大値でラベルを決める(argmax)。
  - 誤爆しやすい行種(電話番号・住所・小計・税対象額・商品行など)は
    'other' 行きのディストラクタプロトタイプとして明示的に持つ。
  - best スコアが閾値未満なら 'other'。
  - 数字は '0' に正規化してから埋め込む(金額・電話番号の具体値ノイズを消し、
    「ラベル部分の意味」で比較するため)。プロトタイプ側も同じ正規化を通す。

プレフィックス: Ruri v3 の 1+3 プレフィックス方式のうち、
  意味類似 = "" / トピック分類 = "トピック: " の2案を切替可能(デフォルト "")。

ラベル→値抽出(単純版):
  - 金額系(total/deposit/change/tax): ラベル行内の ¥付き数値を優先して末尾の数値を取る。
    行内で完結するため「現金」行の次行の電話番号を拾う類の複数行誤マッチは構造的に起きない。
  - tax はラベル行が複数ある場合、税率なし行を優先し、無ければ税率ごとに合算
    (extract_fields と同じ方針の簡易版)。
  - datetime 行から日付・時刻を正規表現で取る。
  - store_branch はラベル行末尾の「〜店」を取る。
"""
import re
import unicodedata

import numpy as np

MODEL_PATH = 'models/ruri-v3-310m'

# フィールドプロトタイプ(値の数字は正規化されるので具体値に意味はない)
PROTOTYPES = {
    'total': [
        '合計 ¥494',
        '合 計 ¥1,800',
        '合計金額 ¥2,503',
        'お買上げ計 ¥1,234',
        'お買上合計 ¥3,150',
        '総合計 ¥5,000',
        'ご請求金額 ¥1,000',
        '合計(税込) ¥980',
        # 総額のみを大きく印字する領収書レイアウト(「¥N.-」)
        '¥2,503.-',
        # 単一税率の領収書で合計が「対象合計金額」としてのみ印字されるケース
        '10%対象合計金額 ¥1,280',
    ],
    'deposit': [
        'お預り ¥1,000',
        'お預かり ¥10,000',
        'お預り合計 ¥10,000',
        '現金 ¥6,000',
        '現金お預り ¥5,000',
        'お預り金額 ¥2,000',
    ],
    'change': [
        'お釣り ¥506',
        'おつり ¥0',
        'お釣 ¥9,506',
        'お釣り銭 ¥494',
    ],
    'tax': [
        '(内消費税等 ¥36)',
        '内消費税額 ¥36',
        '消費税等(8%) ¥21',
        '内税額 10.0% ¥228',
        '(内税) ¥388',
        '10%対象内消費税 ¥290',
        '(内税分消費税 ¥91)',
        '消費税 ¥100',
    ],
    'datetime': [
        '2026年 6月16日(火) 18:53',
        '2026年04月16日(木) 10:29',
        '2026/07/10 20:59',
        '2026.6.9 16:07',
        '2026年7月10日 18時36分',
        '26.07.07 09:19',
        # レジ番号・伝票番号・担当者が後置されるパターン
        '2026年01月05日 12:34 伝票No.1234',
        '26.03.15 10:20 レジ2 責105',
        '領収日付:2026年4月18日 13時05分',
    ],
    'store_branch': [
        '新宿東口店',
        '渋谷駅前店',
        '本町一丁目店',
        'イオンモール北店',
        'セルフ青葉台店',
    ],
    'registration_number': [
        '登録番号 T1234567890123',
        '事業者登録番号 T9876543210987',
        '登録番号:T2345678901234',
    ],
}

# 'other' 行きディストラクタ(誤爆しやすい行種を明示的に受け止める)
DISTRACTORS = [
    '電話：018-853-0502',
    'TEL 0120-134-890',
    '電話番号 0120-000-000',
    # 住所(店ラベルの地名と衝突しないよう「〜店」を含む地名は避ける)
    '東京都新宿区西新宿1-2-3 ビル2F',
    '大阪府大阪市北区梅田1-2-3',
    '小計 ¥268',
    '小計(税抜8%) ¥268',
    '(税率8%対象 ¥289)',
    '(8%対象 ¥494)',
    '10%対象 ¥3,150',
    '10%対象額 ¥3,650',
    '8%対象額 ¥750',
    '(商品代金 ¥288)',
    '値引額 -20',
    '(値引合計 -20)',
    'ハウスWウコンのカ100ml',
    'コーヒー ¥130',
    '牛乳 1点 ¥208',
    '入浴料 大人1名 ¥950',
    '施設利用料 ¥1,200',
    '会員入会金 ¥1,500',
    'PayPay支払 ¥289',
    'クレジット支払 ¥5,240',
    'ポイント残高 120P',
    '点数 2個',
    '上記正に領収いたしました',
    'ありがとうございました。またお越しくださいませ',
    '軽印は軽減税率対象商品です。',
    '#1227620260416102935416120216895',
    '伝票番号 260-416-243-1757',
    'レジ#2 担当:佐々木',
    '領収書',
    '株式会社ローソン',
]

DEFAULT_THRESHOLD = 0.80


def normalize_line(text):
    """NFKC + 数字を'0'へ潰す(埋め込み用)。"""
    t = unicodedata.normalize('NFKC', text)
    t = re.sub(r'\d', '0', t)
    return t


# ---------------- 値抽出(単純版) ----------------
_NUM_RE = re.compile(r'[¥\\]\s*(\d{1,3}(?:[,，.]\s*\d{3})+|\d+)|(\d{1,3}(?:[,，.]\s*\d{3})+|\d+)')


def extract_amount(line):
    """行内の金額を取る。¥付き数値を優先、無ければ最後の数値。"""
    line = unicodedata.normalize('NFKC', line)
    yen, bare = [], []
    for m in _NUM_RE.finditer(line):
        if m.group(1) is not None:
            yen.append(m.group(1))
        else:
            bare.append(m.group(2))
    pick = (yen or bare)
    if not pick:
        return None
    return int(re.sub(r'[,，.\s]', '', pick[-1]))


_DATE_PATTERNS = [
    r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日',
    r'(?<!\d)(\d{4})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})',
    r'(?<![\d.])(\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})',
]


def extract_date(line):
    line = unicodedata.normalize('NFKC', line)
    for pat in _DATE_PATTERNS:
        m = re.search(pat, line)
        if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
            y = int(m.group(1))
            if y < 100:
                y += 2000
            return f"{y}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def extract_time(line):
    line = unicodedata.normalize('NFKC', line)
    m = re.search(r'(\d{1,2})\s*[::]\s*(\d{2})(?!\d)', line) \
        or re.search(r'(\d{1,2})\s*時\s*(\d{1,2})\s*分', line)
    if m and int(m.group(1)) < 24:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    return None


def extract_branch(line):
    compact = re.sub(r'\s+', '', unicodedata.normalize('NFKC', line))
    if '対象' in compact:
        return None
    # 行末の「〜店」を優先。無ければ行中の最初の「〜店」
    # (「◯◯店 秋田県秋田市」のように住所が後続するケース)。
    # 行中フォールバックはラベラーが store_branch と判定した行にしか適用されない
    # 前提なので、住所行等での誤マッチリスクは低い。
    m = (re.search(r'[ぁ-んァ-ヶ一-龥A-Za-z][ぁ-んァ-ヶ一-龥A-Za-z0-9ー\-]*店$', compact)
         or re.search(r'[ぁ-んァ-ヶ一-龥A-Za-z][ぁ-んァ-ヶ一-龥A-Za-z0-9ー\-]*?店', compact))
    if m:
        return re.sub(r'(?<=[ァ-ヶ])-(?=[ァ-ヶ])', 'ー', m.group(0))
    return None


def extract_regnum(line):
    m = re.search(r'T\d{13}', re.sub(r'\s', '', unicodedata.normalize('NFKC', line)))
    return m.group() if m else None


_RATE_RE = re.compile(r'(\d{1,2}(?:\.\d+)?)\s*%')


class RuriLineLabeler:
    def __init__(self, model_path=MODEL_PATH, prefix='', threshold=DEFAULT_THRESHOLD,
                 device='cpu', dims=None):
        """prefix: '' (意味類似) or 'トピック: ' (トピック分類)。
        dims: 先頭k次元だけ使う場合の k(None=768全次元)。"""
        import time
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        self.model = SentenceTransformer(model_path, device=device)
        self.load_sec = time.time() - t0
        self.prefix = prefix
        self.threshold = threshold
        self.dims = dims

        self.proto_texts = []
        self.proto_labels = []
        for label, sents in PROTOTYPES.items():
            for s in sents:
                self.proto_texts.append(s)
                self.proto_labels.append(label)
        for s in DISTRACTORS:
            self.proto_texts.append(s)
            self.proto_labels.append('other')
        self.proto_emb = self._encode(self.proto_texts)

    def _encode(self, texts):
        inputs = [self.prefix + normalize_line(t) for t in texts]
        emb = self.model.encode(inputs, convert_to_numpy=True,
                                normalize_embeddings=True, batch_size=32)
        return self._reduce(emb)

    def _reduce(self, emb):
        if self.dims is not None:
            emb = emb[:, :self.dims]
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def encode_lines(self, lines):
        """行埋め込みのみ(計測・キャッシュ用)。"""
        return self._encode(lines)

    def label_from_embeddings(self, line_emb, threshold=None):
        """各行の (label, score) を返す。scoreは最良プロトタイプとのcos類似度。"""
        threshold = self.threshold if threshold is None else threshold
        sims = line_emb @ self.proto_emb.T  # (n_lines, n_protos)
        best_idx = sims.argmax(axis=1)
        out = []
        for i, j in enumerate(best_idx):
            score = float(sims[i, j])
            label = self.proto_labels[j]
            if score < threshold:
                label = 'other'
            out.append((label, score))
        return out

    def label_lines(self, lines, threshold=None):
        return self.label_from_embeddings(self.encode_lines(lines), threshold)

    # ---------------- ラベル→フィールド値 ----------------
    def extract_from_labels(self, lines, labels):
        """labels: label_lines() の出力。fields dict を返す。"""
        by_field = {}
        for line, (label, score) in zip(lines, labels):
            if label != 'other':
                # ラベル文字(かな漢字)を含む行を裸数字行より優先する。
                # 劣化画像では裸金額の誤読行(「¥1.299」等)が高スコアになりやすく、
                # ラベル付きの行(「領収金額 ¥1,200」)の方が値の信頼性が高いため。
                has_label_chars = bool(re.search(r'[ぁ-んァ-ヶ一-龥]', line))
                by_field.setdefault(label, []).append((has_label_chars, score, line))
        for v in by_field.values():
            v.sort(reverse=True)
        by_field = {k: [(s, l) for _, s, l in v] for k, v in by_field.items()}

        result = {}
        for field in ('total', 'deposit', 'change'):
            for score, line in by_field.get(field, []):
                amt = extract_amount(line)
                if amt is not None:
                    result[field] = amt
                    break
        # tax: 税率なし行優先、無ければ税率ごとに1件ずつ合算
        no_rate, by_rate = None, {}
        for score, line in by_field.get('tax', []):
            amt = extract_amount(line)
            if amt is None:
                continue
            rates = [r for r in _RATE_RE.findall(unicodedata.normalize('NFKC', line))]
            if not rates:
                if no_rate is None:
                    no_rate = amt
            else:
                by_rate.setdefault(float(rates[0]), amt)
        if no_rate is not None:
            result['tax'] = no_rate
        elif by_rate:
            result['tax'] = sum(by_rate.values())

        for score, line in by_field.get('datetime', []):
            if 'date' not in result:
                d = extract_date(line)
                if d:
                    result['date'] = d
            if 'time' not in result:
                t = extract_time(line)
                if t:
                    result['time'] = t
            if 'date' in result and 'time' in result:
                break
        for score, line in by_field.get('store_branch', []):
            b = extract_branch(line)
            if b:
                result['store_branch'] = b
                break
        for score, line in by_field.get('registration_number', []):
            r = extract_regnum(line)
            if r:
                result['registration_number'] = r
                break
        return result

    def extract(self, lines, threshold=None):
        labels = self.label_lines(lines, threshold)
        return self.extract_from_labels(lines, labels)
