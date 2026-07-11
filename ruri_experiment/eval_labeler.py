#!/usr/bin/env python3
"""
ruri-v3-310m 行ラベリングの全評価データ一括評価。

スイート: real(実写9枚) / synth(合成20枚) / external clean/light/medium/heavy(各10枚)
OCR: OnnxReceiptOCR FP32(結果は ruri_experiment/ocr_cache.json にキャッシュ)

計測:
  (A) 各GTフィールドについて「正解値を含む行」を正しいラベルに割り当てられた率
  (B) 現行 extract_fields のNGのうち、行ラベリング→単純値抽出で回収できた件数
  (C) 現行正解を行ラベリング抽出が壊す件数(誤値/未検出の別)
  + 310m のロード時間・1画像あたり行埋め込み時間・ピークRSS
  + 次元削減(先頭k次元)と閾値の感度

実行:
  .venv/bin/python ruri_experiment/eval_labeler.py --build-cache   # OCRキャッシュ作成(初回)
  .venv/bin/python ruri_experiment/eval_labeler.py                 # 評価本体
  .venv/bin/python ruri_experiment/eval_labeler.py --prefix topic  # トピックプレフィックス比較
"""
import argparse
import json
import os
import re
import resource
import sys
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from receipt_ocr_paddle import extract_fields, paddle_result_to_text  # noqa: E402
from benchmark_models import GROUND_TRUTHS  # noqa: E402
from eval_external import build_targets, gt_from_source, is_match  # noqa: E402
from ruri_experiment.line_labeler import (  # noqa: E402
    RuriLineLabeler, PROTOTYPES, DEFAULT_THRESHOLD)

CACHE_PATH = os.path.join(ROOT, 'ruri_experiment', 'ocr_cache.json')

# ラベラーが扱うフィールド(store_brand は辞書照合の領分なので対象外)
LABELER_FIELDS = {'total', 'deposit', 'change', 'tax', 'date', 'time',
                  'store_branch', 'registration_number'}
FIELD_TO_LABEL = {'date': 'datetime', 'time': 'datetime'}


# ---------------- データセット列挙 ----------------
def load_items():
    """[{suite, key, path, gt, extra_brands}]"""
    items = []
    for name, gt in GROUND_TRUTHS.items():
        items.append({'suite': 'real', 'key': name,
                      'path': os.path.join(ROOT, name), 'gt': gt,
                      'extra_brands': None})
    with open(os.path.join(ROOT, 'synth', 'gt.json'), encoding='utf-8') as f:
        sgts = json.load(f)
    synth_brands = sorted({g['store_brand'] for g in sgts.values()
                           if 'store_brand' in g})
    for name, gt in sorted(sgts.items()):
        items.append({'suite': 'synth', 'key': name,
                      'path': os.path.join(ROOT, 'synth', 'images', name),
                      'gt': gt, 'extra_brands': synth_brands})
    targets, gts = build_targets()
    for case_id in sorted(targets):
        for cond in ('clean', 'light', 'medium', 'heavy'):
            path = targets[case_id].get(cond)
            if path is None:
                continue
            items.append({'suite': f'external-{cond}', 'key': f'{case_id}/{cond}',
                          'path': path, 'gt': gts[case_id], 'extra_brands': None})
    return items


# ---------------- OCRキャッシュ ----------------
def build_cache(items):
    import cv2
    from onnx_receipt_ocr import OnnxReceiptOCR
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            cache = json.load(f)
    missing = [it for it in items if it['key'] not in cache]
    if not missing:
        return cache
    ocr = OnnxReceiptOCR()  # FP32
    for i, it in enumerate(missing):
        img = cv2.imread(it['path'])
        res = ocr.predict(img)
        text = paddle_result_to_text(res)
        cache[it['key']] = {'text': text}
        print(f"  OCR [{i + 1}/{len(missing)}] {it['key']} "
              f"({len(text.splitlines())} lines)", flush=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    return cache


# ---------------- GT値が行に含まれるか ----------------
def _compact(s):
    return re.sub(r'[,，.\s]', '', unicodedata.normalize('NFKC', s))


def value_in_line(field, value, line):
    if field in ('total', 'deposit', 'change', 'tax'):
        values = value if isinstance(value, list) else [value]
        comp = _compact(line)
        return any(re.search(r'(?<!\d)' + str(v) + r'(?!\d)', comp) for v in values)
    if field == 'date':
        y, m, d = value.split('-')
        comp = re.sub(r'\s', '', unicodedata.normalize('NFKC', line))
        pats = [rf'{y}年0?{int(m)}月0?{int(d)}日',
                rf'{y}[/.\-]0?{int(m)}[/.\-]0?{int(d)}',
                rf'{y[2:]}\.0?{int(m)}\.0?{int(d)}']
        return any(re.search(p, comp) for p in pats)
    if field == 'time':
        h, mi = value.split(':')
        comp = re.sub(r'\s', '', unicodedata.normalize('NFKC', line))
        return bool(re.search(rf'0?{int(h)}[::]{mi}(?!\d)', comp)
                    or re.search(rf'0?{int(h)}時0?{int(mi)}分', comp))
    if field in ('store_branch', 'registration_number'):
        return _compact(str(value)) in _compact(line)
    return False


def labeler_value_match(suite, field, got, expected):
    """ラベラー抽出値の正誤判定(store_branch は末尾一致で寛容に)"""
    if got is None:
        return False
    if field == 'store_branch':
        exp = expected if isinstance(expected, str) else str(expected)
        return got == exp or got.endswith(exp) or exp.endswith(got)
    if suite.startswith('external'):
        return is_match(field, got, expected)
    return got == expected


def current_match(suite, field, got, expected):
    if suite.startswith('external'):
        return is_match(field, got, expected)
    return got == expected


# ---------------- 評価本体 ----------------
def suite_order(items):
    seen = []
    for it in items:
        if it['suite'] not in seen:
            seen.append(it['suite'])
    return seen


def evaluate(items, cache, labeler, threshold, verbose=False,
             line_embs=None, dims=None):
    """全メトリクスを計算して dict で返す。line_embs を渡すと再埋め込みしない。"""
    suites = suite_order(items)
    A = {s: {'ok': 0, 'eligible': 0, 'absent': 0} for s in suites}
    B = {s: {'recovered': 0, 'ng': 0, 'ng_missing': 0, 'ng_wrong': 0,
             'recovered_missing': 0, 'recovered_wrong': 0} for s in suites}
    C = {s: {'broken_wrong': 0, 'broken_missing': 0, 'ok': 0} for s in suites}
    b_examples, c_examples, a_fail_examples = [], [], []
    embed_times, line_counts = [], []
    out_embs = {}

    for it in items:
        suite, key, gt = it['suite'], it['key'], it['gt']
        text = cache[key]['text']
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            lines = ['']
        if line_embs is not None and key in line_embs:
            emb = line_embs[key]
        else:
            t0 = time.time()
            emb = labeler.encode_lines(lines)
            embed_times.append(time.time() - t0)
            line_counts.append(len(lines))
        out_embs[key] = emb
        emb_r = emb
        proto = labeler.proto_emb
        if dims is not None:
            emb_r = emb[:, :dims]
            emb_r = emb_r / np.linalg.norm(emb_r, axis=1, keepdims=True)
            proto = labeler.proto_emb[:, :dims]
            proto = proto / np.linalg.norm(proto, axis=1, keepdims=True)
        sims = emb_r @ proto.T
        best = sims.argmax(axis=1)
        labels = []
        for i, j in enumerate(best):
            score = float(sims[i, j])
            lab = labeler.proto_labels[j] if score >= threshold else 'other'
            labels.append((lab, score))

        current = extract_fields(text, extra_brands=it['extra_brands'])
        lab_fields = labeler.extract_from_labels(lines, labels)

        for field, expected in gt.items():
            if field not in LABELER_FIELDS:
                continue
            want_label = FIELD_TO_LABEL.get(field, field)
            # --- (A) 正解値を含む行のラベリング成否 ---
            cand = [i for i, l in enumerate(lines)
                    if value_in_line(field, expected, l)]
            if not cand:
                A[suite]['absent'] += 1
            else:
                A[suite]['eligible'] += 1
                hit = any(labels[i][0] == want_label for i in cand)
                A[suite]['ok'] += hit
                if not hit and verbose:
                    got_labels = [(lines[i], labels[i]) for i in cand]
                    a_fail_examples.append((suite, key, field, got_labels))
            # --- (B)(C) 現行との比較 ---
            cur_got = current.get(field)
            cur_ok = current_match(suite, field, cur_got, expected)
            lab_got = lab_fields.get(field)
            lab_ok = labeler_value_match(suite, field, lab_got, expected)
            if cur_ok:
                C[suite]['ok'] += 1
                if not lab_ok:
                    kind = 'broken_missing' if lab_got is None else 'broken_wrong'
                    C[suite][kind] += 1
                    c_examples.append((suite, key, field, expected, lab_got))
            else:
                B[suite]['ng'] += 1
                kind = 'missing' if cur_got is None else 'wrong'
                B[suite][f'ng_{kind}'] += 1
                if lab_ok:
                    B[suite]['recovered'] += 1
                    B[suite][f'recovered_{kind}'] += 1
                    b_examples.append((suite, key, field, expected, cur_got, lab_got))
    return {'A': A, 'B': B, 'C': C,
            'b_examples': b_examples, 'c_examples': c_examples,
            'a_fail_examples': a_fail_examples,
            'embed_times': embed_times, 'line_counts': line_counts,
            'line_embs': out_embs}


def print_report(res, suites, threshold, show_examples=True):
    A, B, C = res['A'], res['B'], res['C']
    print(f"\n=== threshold={threshold} ===")
    print(f"{'suite':<18} {'(A)正ラベル率':>16} {'(値が行に無い)':>10} "
          f"{'(B)回収/NG':>12} {'(C)破壊/現行OK':>12}")
    tA = tAe = tAab = tB = tBn = tCb = tCok = 0
    for s in suites:
        a, b, c = A[s], B[s], C[s]
        broken = c['broken_wrong'] + c['broken_missing']
        ar = f"{a['ok']}/{a['eligible']}" if a['eligible'] else "-"
        print(f"{s:<18} {ar:>14} {a['absent']:>10} "
              f"{b['recovered']}/{b['ng']:>8} {broken}/{c['ok']:>8} "
              f"(誤値{c['broken_wrong']}/欠{c['broken_missing']})")
        tA += a['ok']; tAe += a['eligible']; tAab += a['absent']
        tB += b['recovered']; tBn += b['ng']
        tCb += broken; tCok += c['ok']
    print(f"{'TOTAL':<18} {f'{tA}/{tAe}':>14} {tAab:>10} "
          f"{tB}/{tBn:>8} {tCb}/{tCok:>8}")
    if show_examples:
        if res['b_examples']:
            print("\n--- (B) 回収できたケース ---")
            for s, k, f, exp, cur, lab in res['b_examples']:
                print(f"  [{s}] {k} {f}: 現行={cur} → ラベラー={lab} (GT={exp})")
        if res['c_examples']:
            print("\n--- (C) 壊したケース ---")
            for s, k, f, exp, lab in res['c_examples']:
                print(f"  [{s}] {k} {f}: GT={exp} → ラベラー={lab}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-cache', action='store_true')
    ap.add_argument('--prefix', choices=['none', 'topic'], default='none')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--sweep', action='store_true', help='閾値・次元の感度分析')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--model', default='models/ruri-v3-310m',
                    help='モデルパス (default: models/ruri-v3-310m)')
    args = ap.parse_args()

    items = load_items()
    if args.build_cache:
        build_cache(items)
        print("cache done:", CACHE_PATH)
        return

    with open(CACHE_PATH, encoding='utf-8') as f:
        cache = json.load(f)
    missing = [it['key'] for it in items if it['key'] not in cache]
    if missing:
        print(f"cache に {len(missing)} 件不足 (--build-cache を先に実行)。", missing[:5])
        return

    prefix = '' if args.prefix == 'none' else 'トピック: '
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    labeler = RuriLineLabeler(model_path=args.model, prefix=prefix)
    rss_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    n_protos = len(labeler.proto_texts)
    print(f"model load: {labeler.load_sec:.2f}s / prototypes: {n_protos} "
          f"({ {k: len(v) for k, v in PROTOTYPES.items()} } + other)")

    suites = suite_order(items)
    print(f"\n=== モデル: {args.model} ===")
    res = evaluate(items, cache, labeler, args.threshold, verbose=args.verbose)
    print_report(res, suites, args.threshold)

    if args.verbose and res['a_fail_examples']:
        print("\n--- (A) 失敗例(正解値を含む行に付いたラベル) ---")
        for s, k, f, got in res['a_fail_examples']:
            for line, (lab, sc) in got:
                print(f"  [{s}] {k} {f}: '{line}' → {lab} ({sc:.3f})")

    # ---- 性能 ----
    et, lc = res['embed_times'], res['line_counts']
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\n=== 310m CPU実測 ===")
    print(f"モデルロード: {labeler.load_sec:.2f}s")
    print(f"行埋め込み: mean {np.mean(et)*1000:.0f}ms / median {np.median(et)*1000:.0f}ms "
          f"/ max {np.max(et)*1000:.0f}ms per image "
          f"(平均 {np.mean(lc):.1f} 行/画像, 計 {len(et)} 画像)")
    print(f"1行あたり: {sum(et)/sum(lc)*1000:.1f}ms")
    print(f"RSS: start {rss0:.0f}MB → load後 {rss_load:.0f}MB → 終了 {rss_end:.0f}MB (peak)")

    if args.sweep:
        embs = res['line_embs']
        print("\n=== 閾値スイープ ===")
        for th in (0.70, 0.75, 0.80, 0.85, 0.90):
            r = evaluate(items, cache, labeler, th, line_embs=embs)
            print_report(r, suites, th, show_examples=False)
        print("\n=== 次元削減(先頭k次元)スイープ (threshold=%.2f) ===" % args.threshold)
        for k in (768, 384, 256, 128, 64, 32):
            r = evaluate(items, cache, labeler, args.threshold,
                         line_embs=embs, dims=(None if k == 768 else k))
            A = r['A']
            tA = sum(a['ok'] for a in A.values())
            tAe = sum(a['eligible'] for a in A.values())
            tB = sum(b['recovered'] for b in r['B'].values())
            tBn = sum(b['ng'] for b in r['B'].values())
            tC = sum(c['broken_wrong'] + c['broken_missing'] for c in r['C'].values())
            tCok = sum(c['ok'] for c in r['C'].values())
            print(f"  k={k:<4} A={tA}/{tAe}  B回収={tB}/{tBn}  C破壊={tC}/{tCok}")


if __name__ == '__main__':
    main()
