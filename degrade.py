#!/usr/bin/env python3
"""
クリーン合成レシート(K10124 alpha10)を「スマホ撮影風」に劣化合成するスクリプト。
OpenCV + numpy のみ使用。乱数は --seed と (ファイル名, 強度) から決定論的に導出され、
実行順・対象サブセットに依らず同じ出力が再現される。

劣化パイプライン:
  1. 感熱紙風の印字掠れ(低周波の濃度ムラ + サーマルヘッド縦スジ)
  2. 紙の暖色化(白→わずかにクリーム色)+ 紙のカール陰影(左右端が暗い)
  3. プロシージャル背景(木目デスク / 布 / 単色マット)に一回り小さく配置
  4. 透視変換(斜め撮り)+ 回転(±3度、medium/heavyはたまに90度回転)+ 落ち影
  5. 照明ムラ(方向性グラデーション)+ 帯状の影
  6. 軽いモーションブラー + ガウスノイズ + JPEG再圧縮

実行:
  .venv/bin/python degrade.py                 # 3強度すべて生成
  .venv/bin/python degrade.py --strength medium --seed 42

出力:
  external_data/K10124__japan-ocr-mini-benchmark/degraded/<元stem>_<強度>.jpg
  external_data/K10124__japan-ocr-mini-benchmark/degraded/manifest.json
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import cv2
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'external_data', 'K10124__japan-ocr-mini-benchmark')
ALPHA10 = os.path.join(BASE, 'alpha10')
OUT_DIR = os.path.join(BASE, 'degraded')

# 強度プリセット。値は目視調整済み(人間が読める範囲を維持すること)
STRENGTHS = {
    #        透視ジッタ 回転θ  90度率  掠れ   縦スジ  カール  影    照明ムラ  帯影   ブレ  ノイズσ  JPEG品質
    'light':  dict(persp=0.010, rot=1.5, rot90=0.00, fade=0.22, streak=0.08, curl=0.06,
                   shadow=0.15, illum=0.10, band=0.10, blur=0, noise=2.5, jpeg=88),
    'medium': dict(persp=0.028, rot=3.0, rot90=0.20, fade=0.40, streak=0.15, curl=0.10,
                   shadow=0.28, illum=0.20, band=0.22, blur=5, noise=5.0, jpeg=72),
    'heavy':  dict(persp=0.048, rot=3.0, rot90=0.30, fade=0.55, streak=0.24, curl=0.14,
                   shadow=0.40, illum=0.28, band=0.32, blur=7, noise=9.0, jpeg=55),
}


# ---------------- ユーティリティ ----------------
def lowfreq(rng, h, w, cell):
    """cellピクセル格子の乱数を拡大した低周波ノイズマップ [0,1]"""
    gh, gw = max(h // cell, 2) + 2, max(w // cell, 2) + 2
    g = rng.random((gh, gw)).astype(np.float32)
    m = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    # NORM_MINMAXは丸め誤差で範囲を僅かに外れることがあり、後段の ** 演算でNaNになるためクリップ
    return np.clip(cv2.normalize(m, None, 0.0, 1.0, cv2.NORM_MINMAX), 0.0, 1.0)


# ---------------- 1. 感熱紙風の掠れ ----------------
def thermal_fade(img, rng, fade, streak):
    """印字(暗部)だけを低周波マップ+縦スジで薄くする。白地は不変。"""
    h, w = img.shape[:2]
    m = 1.0 - fade * lowfreq(rng, h, w, 48) ** 1.5
    col = rng.random(max(w // 6, 2)).astype(np.float32).reshape(1, -1)
    col = cv2.resize(col, (w, 1), interpolation=cv2.INTER_LINEAR)[0]
    m = m * (1.0 - streak * col)[None, :]
    ink = (255.0 - img.astype(np.float32)) * np.clip(m, 0.2, 1.0)[..., None]
    return np.clip(255.0 - ink, 0, 255)


# ---------------- 2. 紙の色味・カール陰影 ----------------
def paper_tone(img, rng, curl):
    tint = np.array([rng.uniform(0.90, 0.95),   # B: 下げて暖色(クリーム)に
                     rng.uniform(0.96, 0.99),   # G
                     1.0], dtype=np.float32)    # R
    img = img * tint[None, None, :]
    w = img.shape[1]
    shade = 1.0 - curl * np.abs(np.linspace(-1, 1, w, dtype=np.float32)) ** 1.8
    return img * shade[None, :, None]


# ---------------- 3. プロシージャル背景 ----------------
def bg_wood(rng, h, w):
    palette = np.array([[60, 82, 122], [46, 66, 100], [72, 96, 134], [38, 54, 76]],
                       dtype=np.float32)  # BGR系ブラウン
    base = palette[rng.integers(len(palette))]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    warp = 2.5 * lowfreq(rng, h, w, 90)           # 木目のうねり
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (yy * rng.uniform(6, 13) + warp))
    tex = 0.78 + 0.22 * stripes + 0.10 * (lowfreq(rng, h, w, 30) - 0.5)
    img = base[None, None, :] * tex[..., None]
    img += rng.normal(0, 3.0, (h, w, 1)).astype(np.float32)
    return cv2.blur(np.clip(img, 0, 255), (9, 1))  # 横方向に流して木目らしく


def bg_cloth(rng, h, w):
    palette = np.array([[108, 104, 96], [88, 78, 66], [122, 120, 116], [96, 84, 62]],
                       dtype=np.float32)
    base = palette[rng.integers(len(palette))]
    period = rng.uniform(2.0, 3.5)
    weave = (np.sin(np.arange(w, dtype=np.float32) * np.pi / period)[None, :]
             + np.sin(np.arange(h, dtype=np.float32) * np.pi / period)[:, None])
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.0)
    tex = 1.0 + 0.05 * weave + 0.06 * n + 0.16 * (lowfreq(rng, h, w, 60) - 0.5)
    return np.clip(base[None, None, :] * tex[..., None], 0, 255)


def bg_plain(rng, h, w):
    palette = np.array([[150, 148, 144], [96, 96, 96], [70, 86, 96], [130, 120, 104]],
                       dtype=np.float32)
    base = palette[rng.integers(len(palette))]
    tex = 1.0 + 0.10 * (lowfreq(rng, h, w, 80) - 0.5)
    img = base[None, None, :] * tex[..., None]
    img += rng.normal(0, 2.0, (h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 255)


BG_FUNCS = [bg_wood, bg_cloth, bg_plain]


# ---------------- 4. 配置(透視・回転・落ち影) ----------------
def compose(receipt, rng, p):
    h, w = receipt.shape[:2]
    mw, mh = int(w * 0.20) + 8, int(h * 0.10) + 8   # 縦長レシートなので上下は控えめ
    H, W = h + 2 * mh, w + 2 * mw
    bg_kind = int(rng.integers(len(BG_FUNCS)))
    bg = BG_FUNCS[bg_kind](rng, H, W)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * [p['persp'] * w, p['persp'] * h]
    dst = src + np.float32([mw, mh]) + jitter
    ang = np.deg2rad(rng.uniform(-p['rot'], p['rot']))
    c, s = np.cos(ang), np.sin(ang)
    center = np.float32([W / 2, H / 2])
    dst = ((dst - center) @ np.float32([[c, -s], [s, c]]).T + center).astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(receipt, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    maskf = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0)[..., None]

    # 落ち影: マスクを右下にずらしてぼかし、レシート外のみ暗くする
    dy, dx = int(H * 0.012) + 4, int(W * 0.010) + 3
    sh = np.roll(mask, (dy, dx), axis=(0, 1)).astype(np.float32) / 255.0
    sh = cv2.GaussianBlur(sh, (0, 0), max(H, W) * 0.012)
    bg = bg * (1.0 - p['shadow'] * sh * (1.0 - maskf[..., 0]))[..., None]

    return bg * (1.0 - maskf) + warped * maskf, bg_kind


# ---------------- 5. 照明ムラ・帯影 ----------------
def illumination(img, rng, strength, band):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    proj = xx * np.cos(th) + yy * np.sin(th)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)
    grad = 1.0 + strength * (proj - 0.5)          # 片側 1-s/2 〜 反対側 1+s/2
    img = img * grad[..., None]
    if band > 0:
        th2 = rng.uniform(0, np.pi)
        proj2 = xx * np.cos(th2) + yy * np.sin(th2)
        proj2 = (proj2 - proj2.min()) / (np.ptp(proj2) + 1e-6)
        dist = np.abs(proj2 - rng.uniform(0.25, 0.75))
        bw = rng.uniform(0.07, 0.16)
        img = img * (1.0 - band * np.exp(-(dist / bw) ** 2))[..., None]
    return img


# ---------------- 6. ブレ・ノイズ・JPEG ----------------
def motion_blur(img, rng, k):
    if k < 3:
        return img
    kern = np.zeros((k, k), np.float32)
    th = rng.uniform(0, np.pi)
    r = (k - 1) / 2
    x0, y0 = int(round(r - r * np.cos(th))), int(round(r - r * np.sin(th)))
    x1, y1 = int(round(r + r * np.cos(th))), int(round(r + r * np.sin(th)))
    cv2.line(kern, (x0, y0), (x1, y1), 1.0, 1)
    kern /= max(kern.sum(), 1e-6)
    return cv2.filter2D(img, -1, kern)


def degrade_one(img_bgr, rng, p):
    """クリーンレシート(BGR uint8) → スマホ撮影風(BGR uint8) + 適用パラメータ"""
    x = thermal_fade(img_bgr, rng, p['fade'], p['streak'])
    x = paper_tone(x, rng, p['curl'])
    x, bg_kind = compose(x, rng, p)
    rot90 = 0
    if rng.random() < p['rot90']:
        rot90 = int(rng.choice([1, 3]))           # 90度 or 270度
        x = np.ascontiguousarray(np.rot90(x, rot90))
    x = illumination(x, rng, p['illum'], p['band'])
    x = motion_blur(x, rng, p['blur'])
    x = x + rng.normal(0, p['noise'], x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x, {'bg': BG_FUNCS[bg_kind].__name__, 'rot90_applied': rot90 * 90}


def rng_for(seed, name, strength):
    """(グローバルseed, ファイル名, 強度) から決定論的にRNGを作る"""
    digest = int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)
    si = list(STRENGTHS).index(strength)
    return np.random.default_rng(np.random.SeedSequence([seed, digest, si]))


def load_case_index():
    """alpha10_manifest.csv から image_path → (case_id, source_json_path) を引く"""
    index = {}
    with open(os.path.join(ALPHA10, 'alpha10_manifest.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            index[os.path.basename(row['image_path'])] = (
                row['case_id'], row['source_json_path'])
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--strength', choices=list(STRENGTHS), default=None,
                    help='指定時はその強度のみ生成(既定: 3強度すべて)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    case_index = load_case_index()
    strengths = [args.strength] if args.strength else list(STRENGTHS)
    images = sorted(glob.glob(os.path.join(ALPHA10, 'images', '*.png')))

    entries = []
    for path in images:
        name = os.path.basename(path)
        img = cv2.imread(path)
        case_id, src_json = case_index.get(name, (None, None))
        for st in strengths:
            rng = rng_for(args.seed, name, st)
            out, applied = degrade_one(img, rng, STRENGTHS[st])
            out_name = f"{os.path.splitext(name)[0]}_{st}.jpg"
            ok, buf = cv2.imencode('.jpg', out,
                                   [cv2.IMWRITE_JPEG_QUALITY, STRENGTHS[st]['jpeg']])
            assert ok
            with open(os.path.join(OUT_DIR, out_name), 'wb') as f:
                f.write(buf.tobytes())
            entries.append({
                'output': out_name,
                'strength': st,
                'case_id': case_id,
                'source_image': f'alpha10/images/{name}',
                'source_json': f'alpha10/{src_json}' if src_json else None,
                **applied,
            })
            print(f"{out_name}  ({out.shape[1]}x{out.shape[0]}, bg={applied['bg']},"
                  f" rot90={applied['rot90_applied']})")

    manifest = {
        'generator': 'degrade.py',
        'seed': args.seed,
        'strength_params': {k: STRENGTHS[k] for k in strengths},
        'count': len(entries),
        'entries': entries,
    }
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n{len(entries)}枚生成 → {OUT_DIR}/manifest.json")


if __name__ == '__main__':
    main()
