#!/usr/bin/env python3
"""
日本語レシート合成生成器 — 評価データ量産用。

PIL(Pillow)でレシート画像をクリーン描画し、GROUND_TRUTHS と同一スキーマの
正解値を synth/gt.json に書き出す。店名はすべて架空。

レイアウトテンプレート4種:
    conv    コンビニ風   (内税・税率8/10%混在・現金/QR/クレジット)
    super   スーパー風   (外税・小計+消費税→合計・現金多め)
    dining  飲食店風     (内税8%・大フォント合計・クレジット/現金)
    service サービス業風 (クリーニング。冒頭に金額大書き・内税額10.0%・時分表記)

表記ゆれ(意図的に混在):
    日付   2026年6月16日 / 2026年06月16日(火) / 2026/06/16 / 26.06.16
    時刻   18:53 / 18時53分
    合計   合計 / 合 計 / 合計金額 / 大フォント
    税     内消費税等 / (内税) / 内税額10.0%を含む / 消費税等(8%) 外税
    支払   現金(お預り+お釣り) / クレジット / QR決済(預り釣り無し)
    その他 登録番号T+13桁(乱数・有無あり)、金額カンマ区切り、字間スペース

実行: .venv/bin/python synth/generate_receipts.py [--n 20] [--seed 42]
出力: synth/images/synth_XXX.png, synth/gt.json
"""
import os
import json
import argparse
import random
import datetime

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
GT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gt.json')

FONT_REG = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
FONT_BOLD = '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'

WIDTH = 640
MARGIN = 36
BASE = 22  # 基準フォントサイズ

_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size, index=0)
    return _font_cache[key]


# ============================================================
# 行モデル: (kind, ...) のリストを組み立ててから一括描画
#   ('c',  text, size, bold)            中央寄せ
#   ('l',  text, size, bold)            左寄せ
#   ('r',  text, size, bold)            右寄せ
#   ('lr', left, right, size, bold)     左ラベル+右寄せ金額
#   ('gap', px)                         余白
#   ('dash',)                           破線セパレータ
# ============================================================
def render(lines, path):
    heights = []
    for ln in lines:
        if ln[0] == 'gap':
            heights.append(ln[1])
        elif ln[0] == 'dash':
            heights.append(int(BASE * 1.2))
        else:
            size = ln[3] if ln[0] == 'lr' else ln[2]
            heights.append(int(size * 1.45))
    h = sum(heights) + MARGIN * 2
    img = Image.new('RGB', (WIDTH, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = MARGIN
    for ln, lh in zip(lines, heights):
        kind = ln[0]
        if kind == 'gap':
            pass
        elif kind == 'dash':
            d.text((MARGIN, y), '-' * 38, font=font(BASE), fill=0)
        elif kind == 'lr':
            _, left, right, size, bold = ln
            f = font(size, bold)
            d.text((MARGIN, y), left, font=f, fill=0)
            d.text((WIDTH - MARGIN - f.getlength(right), y), right, font=f, fill=0)
        else:
            _, text, size, bold = ln
            f = font(size, bold)
            if kind == 'c':
                x = (WIDTH - f.getlength(text)) / 2
            elif kind == 'r':
                x = WIDTH - MARGIN - f.getlength(text)
            else:
                x = MARGIN
            d.text((x, y), text, font=f, fill=0)
        y += lh
    img.save(path)


# ============================================================
# 素材(すべて架空)
# ============================================================
BRANDS = {
    'conv': ['マルワ商店', 'サンライズストア', 'エブリデイポート', 'ハピネスプラザ'],
    'super': ['グリーンマート', 'フレッシュいちば', 'スーパーコダマ', 'マルシェタカノ'],
    'dining': ['バーガーキッチンモリ', 'めん処たけや', 'キッチンあおば', 'レストランひまわり'],
    'service': ['しろくまクリーニング', 'クリーニングコスモ', 'ランドリーはなおか'],
}
BRANCHES = ['秋田駅前店', '広面店', '中通二丁目店', '御所野店', '外旭川店',
            '泉中央店', '土崎港店', '仁井田本町店', '手形山崎店', '川尻大川町店']
TOWNS = ['秋田県秋田市山王3-4-12', '秋田県秋田市広面字川崎8-2', '秋田県秋田市泉北1-6-9',
         '秋田県秋田市御野場2-11-5', '秋田県秋田市土崎港中央4-7-1']

ITEMS = {
    'conv': [  # (品名, 税込単価, 税率)
        ('おにぎり ツナマヨ', 138, 8), ('サンドたまご', 268, 8), ('緑茶 525ml', 108, 8),
        ('カフェオレ 240ml', 128, 8), ('チョコビスケット', 158, 8), ('カップめんしょうゆ', 214, 8),
        ('ヨーグルト4個パック', 192, 8), ('ボールペン0.5黒', 110, 10), ('乾電池単3 4本', 498, 10),
        ('ポケットティッシュ', 128, 10),
    ],
    'super': [  # (品名, 税抜単価, 税率)
        ('国産豚こま切れ', 398, 8), ('キャベツ1玉', 158, 8), ('牛乳1000ml', 208, 8),
        ('食パン6枚切', 138, 8), ('たまご10個', 258, 8), ('バナナ', 128, 8),
        ('豆腐3個パック', 88, 8), ('冷凍うどん5食', 248, 8), ('洗濯洗剤詰替', 328, 10),
        ('食品ラップ30cm', 118, 10),
    ],
    'dining': [  # (品名, 税込単価, 税率8=テイクアウト)
        ('チーズバーガー', 480, 8), ('てりやきバーガー', 450, 8), ('フライドポテトM', 330, 8),
        ('ジンジャーエールM', 270, 8), ('チキンナゲット5P', 390, 8), ('シェイクバニラS', 240, 8),
        ('ホットコーヒーM', 280, 8),
    ],
    'service': [  # (品名, 税込単価, 10%)
        ('ワイシャツ', 180, 10), ('ズボン', 550, 10), ('ジャケット', 770, 10),
        ('コート', 1320, 10), ('セーター', 440, 10), ('スカート', 550, 10),
    ],
}


# ============================================================
# 表記ゆれヘルパ
# ============================================================
def yen(n, comma=True):
    return f"¥{n:,}" if comma else f"¥{n}"


def gen_date(rng):
    d0 = datetime.date(2025, 6, 1)
    d = d0 + datetime.timedelta(days=rng.randrange(400))
    wd = '月火水木金土日'[d.weekday()]
    style = rng.choice(['kanji', 'kanji_pad', 'slash', 'dot'])
    if style == 'kanji':
        s = f"{d.year}年{d.month}月{d.day}日({wd})"
    elif style == 'kanji_pad':
        s = f"{d.year}年{d.month:2d}月{d.day:2d}日"
    elif style == 'slash':
        s = f"{d.year}/{d.month:02d}/{d.day:02d}"
    else:
        s = f"{str(d.year)[2:]}.{d.month:02d}.{d.day:02d}"
    return s, d.isoformat()


def gen_time(rng, allow_kanji=True):
    hh, mm = rng.randrange(8, 22), rng.randrange(60)
    if allow_kanji and rng.random() < 0.3:
        s = f"{hh}時{mm:02d}分"
    else:
        s = f"{hh:02d}:{mm:02d}"
    return s, f"{hh:02d}:{mm:02d}"


def gen_regnum(rng):
    return 'T' + ''.join(rng.choice('0123456789') for _ in range(13))


def spaced(s):
    return ' '.join(s)


def total_label(rng):
    return rng.choice(['合計', '合 計', '合計金額'])


def pick_items(rng, pool, n_min=2, n_max=6):
    n = rng.randrange(n_min, n_max + 1)
    picked = rng.sample(pool, min(n, len(pool)))
    return [(name, price, rate, rng.choice([1, 1, 1, 2])) for name, price, rate in picked]


def incl_tax_of(items):
    """内税方式: 税率別の内税額(切り捨て)と合計"""
    total = sum(p * q for _, p, _, q in items)
    tax = 0
    by_rate = {}
    for _, p, r, q in items:
        by_rate[r] = by_rate.get(r, 0) + p * q
    for r, amt in by_rate.items():
        tax += int(amt * r / (100 + r))
    return total, tax, by_rate


def payment_lines(rng, total, gt, size=BASE):
    """支払方法の行を生成し、現金ならGTに deposit/change を追加"""
    method = rng.choice(['cash', 'cash', 'credit', 'qr'])
    lines = []
    if method == 'cash':
        unit = rng.choice([100, 500, 1000, 1000, 5000, 10000])
        deposit = ((total + unit - 1) // unit) * unit
        if deposit == total and rng.random() < 0.7:
            deposit += unit
        change = deposit - total
        dep_label = rng.choice(['お預り', 'お預り合計', 'お預かり'])
        chg_label = rng.choice(['お釣り', 'お 釣'])
        lines.append(('lr', dep_label, yen(deposit), size, False))
        lines.append(('lr', chg_label, yen(change), size, False))
        gt['deposit'] = deposit
        gt['change'] = change
    elif method == 'credit':
        lines.append(('lr', '(クレジット', yen(total) + ')', size, False))
    else:
        pay = rng.choice(['スマホペイ', 'コインペイ', 'QRペイ'])
        lines.append(('lr', f'{pay}支払', yen(total), size, False))
    return lines


# ============================================================
# テンプレート
# ============================================================
def tpl_conv(rng):
    brand = rng.choice(BRANDS['conv'])
    branch = rng.choice(BRANCHES)
    date_s, date_gt = gen_date(rng)
    time_s, time_gt = gen_time(rng, allow_kanji=False)
    items = pick_items(rng, ITEMS['conv'], 2, 6)
    total, tax, by_rate = incl_tax_of(items)

    gt = {'store_brand': brand, 'store_branch': branch,
          'date': date_gt, 'time': time_gt, 'total': total, 'tax': tax}
    lines = [
        ('c', brand, 40, True),
        ('c', branch, 26, False),
    ]
    if rng.random() < 0.85:
        reg = gen_regnum(rng)
        gt['registration_number'] = reg
        sep = rng.choice([':', ';', ' '])
        lines.append(('c', f"登録番号{sep}{reg}", BASE, False))
    lines += [
        ('c', rng.choice(TOWNS), BASE, False),
        ('c', f"電話:018-8{rng.randrange(10,100)}-{rng.randrange(1000,10000)}", BASE, False),
        ('l', f"{date_s} {time_s}", BASE, False),
        ('gap', 8),
        ('c', rng.choice(['領収証', spaced('領収証'), '【領収証】']), 30, True),
        ('gap', 8),
    ]
    for name, price, rate, q in items:
        mark = rng.choice(['※', '軽']) if rate == 8 else ''
        if q == 1:
            lines.append(('lr', name + mark, yen(price), BASE, False))
        else:
            lines.append(('l', name + mark, BASE, False))
            lines.append(('lr', f"  {price} {q}個", yen(price * q), BASE, False))
    lines.append(('dash',))
    big = rng.random() < 0.4
    lines.append(('lr', total_label(rng), yen(total),
                  32 if big else BASE, big))
    tax_style = rng.choice(['uchi_et', 'uchi_paren'])
    if tax_style == 'uchi_et':
        lines.append(('lr', '(内消費税等', yen(tax) + ')', BASE, False))
        for r, amt in sorted(by_rate.items(), reverse=True):
            lines.append(('lr', f"({r}%対象", yen(amt) + ')', BASE, False))
            lines.append(('lr', '(内消費税額', yen(int(amt * r / (100 + r))) + ')', BASE, False))
    else:
        lines.append(('lr', '(内税)', yen(tax), BASE, False))
    lines.append(('lr', '点数', f"{sum(q for *_, q in items)}個", BASE, False))
    lines.append(('l', '上記正に領収いたしました', BASE, False))
    lines += payment_lines(rng, total, gt)
    if any(r == 8 for _, _, r, _ in items):
        lines.append(('l', rng.choice(['※印は軽減税率対象商品です。', '軽印は軽減税率対象商品です。']), BASE, False))
    return lines, gt


def tpl_super(rng):
    brand = rng.choice(BRANDS['super'])
    branch = rng.choice(BRANCHES)
    date_s, date_gt = gen_date(rng)
    time_s, time_gt = gen_time(rng, allow_kanji=False)
    items = pick_items(rng, ITEMS['super'], 3, 7)
    # 外税方式
    by_rate = {}
    for _, p, r, q in items:
        by_rate[r] = by_rate.get(r, 0) + p * q
    subtotal = sum(by_rate.values())
    tax = sum(int(amt * r / 100) for r, amt in by_rate.items())
    total = subtotal + tax

    gt = {'store_brand': brand, 'store_branch': branch,
          'date': date_gt, 'time': time_gt, 'total': total, 'tax': tax}
    lines = [
        ('c', brand, 38, True),
        ('c', branch, 26, False),
        ('c', rng.choice(TOWNS), BASE, False),
    ]
    if rng.random() < 0.85:
        reg = gen_regnum(rng)
        gt['registration_number'] = reg
        lines.append(('c', f"登録番号 {reg}", BASE, False))
    lines += [
        ('l', f"{date_s} {time_s} レジ{rng.randrange(1,9)} 責{rng.randrange(100,999)}", BASE, False),
        ('dash',),
    ]
    for name, price, rate, q in items:
        mark = '※' if rate == 8 else ''
        if q == 1:
            lines.append(('lr', name + mark, f"{price:,}", BASE, False))
        else:
            lines.append(('lr', f"{name}{mark} {q}コX単{price}", f"{price*q:,}", BASE, False))
    lines.append(('dash',))
    lines.append(('lr', rng.choice(['小計', '小 計']), yen(subtotal), BASE, False))
    for r, amt in sorted(by_rate.items(), reverse=True):
        lines.append(('lr', f"消費税等({r}%)", yen(int(amt * r / 100)), BASE, False))
    big = rng.random() < 0.5
    lines.append(('lr', total_label(rng), yen(total), 32 if big else BASE, big))
    lines += payment_lines(rng, total, gt)
    lines.append(('l', '※印は軽減税率(8%)適用商品', BASE, False))
    return lines, gt


def tpl_dining(rng):
    brand = rng.choice(BRANDS['dining'])
    branch = rng.choice(BRANCHES)
    date_s, date_gt = gen_date(rng)
    time_s, time_gt = gen_time(rng, allow_kanji=False)
    items = pick_items(rng, ITEMS['dining'], 2, 5)
    total, tax, by_rate = incl_tax_of(items)

    gt = {'store_brand': brand, 'store_branch': branch,
          'date': date_gt, 'time': time_gt, 'total': total, 'tax': tax}
    lines = [
        ('c', brand, 38, True),
        ('c', branch, 26, False),
        ('c', rng.choice(TOWNS), BASE, False),
        ('c', f"電話:018-8{rng.randrange(10,100)}-{rng.randrange(1000,10000)}", BASE, False),
        ('gap', 6),
        ('c', rng.choice([spaced('領収証'), '領収証']), 32, True),
        ('gap', 6),
        ('l', f"{date_s} {time_s}  No.{rng.randrange(1000,9999)}", BASE, False),
        ('l', f"お客様NO {rng.randrange(100,999):04d}", BASE, False),
    ]
    for name, price, rate, q in items:
        if q == 1:
            lines.append(('lr', f"{name}※", yen(price), BASE, False))
        else:
            lines.append(('lr', f"{name}※  {q}", yen(price * q), BASE, False))
    lines.append(('dash',))
    lines.append(('lr', '小計', yen(total), BASE, False))
    lines.append(('lr', '(内税)', yen(tax), BASE, False))
    lines.append(('lr', rng.choice(['合 計', '合計']), yen(total), 34, True))
    lines += payment_lines(rng, total, gt)
    if rng.random() < 0.85:
        reg = gen_regnum(rng)
        gt['registration_number'] = reg
        lines.append(('gap', 8))
        lines.append(('l', f"登録番号:{reg}", BASE, False))
    lines.append(('l', '「※」は軽減税率対象商品であることを', BASE, False))
    lines.append(('l', '示します。', BASE, False))
    return lines, gt


def tpl_service(rng):
    brand = rng.choice(BRANDS['service'])
    branch = rng.choice(BRANCHES)
    date_s, date_gt = gen_date(rng)
    time_s, time_gt = gen_time(rng, allow_kanji=True)
    items = pick_items(rng, ITEMS['service'], 2, 5)
    total, tax, _ = incl_tax_of(items)

    gt = {'store_brand': brand, 'store_branch': branch,
          'date': date_gt, 'time': time_gt, 'total': total, 'tax': tax}
    lines = [
        ('c', rng.choice(['《領収書》', spaced('領収書')]), 34, True),
        ('r', '様', 26, False),
        ('gap', 10),
        ('c', f"¥{total:,}.-", 44, True),
        ('lr', '10%対象合計金額', yen(total), 26, True),
        ('r', f"(内税額10.0% {yen(tax)}を含む)", BASE, False),
        ('l', '但し、クリーニング代、品代として領収致しました。', BASE, False),
        ('gap', 10),
    ]
    for name, price, rate, q in items:
        lines.append(('lr', f"{name} x{q}", yen(price * q), BASE, False))
    lines.append(('gap', 10))
    lines.append(('l', f"伝票番号:{rng.randrange(10**9, 10**10)}", BASE, False))
    lines.append(('l', f"領収日付: {date_s} {time_s}", BASE, False))
    lines += payment_lines(rng, total, gt)
    lines.append(('l', '《品物・料金に関するお問い合わせ先》', BASE, False))
    if rng.random() < 0.9:
        reg = gen_regnum(rng)
        gt['registration_number'] = reg
        lines.append(('l', f"登録番号 : {reg}", BASE, False))
    lines.append(('l', f"{brand} {branch}", 24, False))
    lines.append(('l', rng.choice(TOWNS), BASE, False))
    lines.append(('l', f"電話:018-8{rng.randrange(10,100)}-{rng.randrange(1000,10000)}", BASE, False))
    return lines, gt


TEMPLATES = [('conv', tpl_conv), ('super', tpl_super),
             ('dining', tpl_dining), ('service', tpl_service)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    gts = {}
    for i in range(args.n):
        tpl_name, tpl = TEMPLATES[i % len(TEMPLATES)]  # 4テンプレートを均等に
        lines, gt = tpl(rng)
        name = f"synth_{i:03d}.png"
        render(lines, os.path.join(OUT_DIR, name))
        gts[name] = gt
        print(f"{name}  [{tpl_name:7s}] {gt['store_brand']} {gt['store_branch']} "
              f"total={gt['total']} tax={gt['tax']}"
              + (f" deposit={gt['deposit']} change={gt['change']}" if 'deposit' in gt else ''))

    with open(GT_PATH, 'w', encoding='utf-8') as f:
        json.dump(gts, f, ensure_ascii=False, indent=2)
    print(f"\n{len(gts)}枚生成 -> {OUT_DIR}\nGT -> {GT_PATH}")


if __name__ == '__main__':
    main()
