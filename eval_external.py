#!/usr/bin/env python3
"""
K10124 alpha10(クリーン原本 + degrade.py生成のスマホ撮影風劣化画像)を
ONNX版パイプライン(OnnxReceiptOCR → paddle_result_to_text → extract_fields)で評価し、
alpha10のsource_json GTと突き合わせるアダプタ。

GTスキーマ(source_json)と extract_fields の出力で対応が取れるフィールドのみ評価する:
  - total               ← totals.total_yen
  - tax                 ← tax_summary.rates(課税対象>0の税率)。レシートには税率別に複数行
                          印字されるが extract_fields は1値しか返さないため、
                          「いずれかの税率の税額 or その合計」に一致すれば正解とする
  - deposit / change    ← payment.cash_received_yen / change_yen(現金払いのみ)
  - registration_number ← invoice_profile.registration_number(印字ありのみ)
  - store_branch        ← store_identity.branch。extract_fieldsは「店名+支店名」を
                          連結で返しうるため、末尾一致(endswith)で正解とする
  ※ 日付・時刻・店名ブランドはGT(source_json)に印字値が含まれないため対象外

実行: .venv/bin/python eval_external.py
"""
import glob
import json
import os
import sys
import time

import cv2

from onnx_receipt_ocr import OnnxReceiptOCR
from receipt_ocr_paddle import extract_fields, paddle_result_to_text  # 編集禁止・import利用のみ

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'external_data', 'K10124__japan-ocr-mini-benchmark')
ALPHA10 = os.path.join(BASE, 'alpha10')
DEGRADED = os.path.join(BASE, 'degraded')

FIELDS = ['total', 'tax', 'deposit', 'change', 'registration_number', 'store_branch']


# ---------------- GT構築 ----------------
def gt_from_source(src):
    """source_json → 評価可能フィールドのGT dict"""
    gt = {'total': src['totals']['total_yen']}
    taxes = [r['tax_amount_yen'] for r in src['tax_summary']['rates']
             if r['taxable_amount_yen'] > 0]
    if taxes:
        gt['tax'] = sorted(set(taxes) | {sum(taxes)})  # 許容値リスト
    pay = src.get('payment', {})
    if 'cash_received_yen' in pay:
        gt['deposit'] = pay['cash_received_yen']
    if 'change_yen' in pay:
        gt['change'] = pay['change_yen']
    inv = src.get('invoice_profile', {})
    if inv.get('registration_number_printed') and inv.get('registration_number'):
        gt['registration_number'] = inv['registration_number']
    branch = src.get('store_identity', {}).get('branch')
    if branch:
        gt['store_branch'] = branch
    return gt


def is_match(field, got, expected):
    if got is None:
        return False
    if field == 'tax':
        return got in expected
    if field == 'store_branch':
        return isinstance(got, str) and (got == expected or got.endswith(expected))
    return got == expected


# ---------------- 評価対象の列挙 ----------------
def build_targets():
    """{case_id: {condition: (image_path, gt)}} を返す"""
    with open(os.path.join(DEGRADED, 'manifest.json'), encoding='utf-8') as f:
        deg_manifest = json.load(f)

    targets = {}
    gts = {}
    for path in sorted(glob.glob(os.path.join(ALPHA10, 'images', '*.png'))):
        stem = os.path.basename(path)
        case_id = stem.split('_')[0]
        src_path = os.path.join(ALPHA10, 'source_json',
                                stem.replace('_clean.png', '_public_source.json'))
        with open(src_path, encoding='utf-8') as f:
            gts[case_id] = gt_from_source(json.load(f))
        targets.setdefault(case_id, {})['clean'] = path
    for e in deg_manifest['entries']:
        targets.setdefault(e['case_id'], {})[e['strength']] = \
            os.path.join(DEGRADED, e['output'])
    return targets, gts


# ---------------- メイン ----------------
def main():
    verbose = '-v' in sys.argv
    conditions = ['clean', 'light', 'medium', 'heavy']
    targets, gts = build_targets()
    ocr = OnnxReceiptOCR()

    # results[case][cond] = {'fields':..., 'ok': set, 'ng': {field: got}}
    results = {}
    t0 = time.time()
    for case_id in sorted(targets):
        gt = gts[case_id]
        results[case_id] = {}
        for cond in conditions:
            path = targets[case_id].get(cond)
            if path is None:
                continue
            img = cv2.imread(path)
            res = ocr.predict(img)
            fields = extract_fields(paddle_result_to_text(res))
            ok, ng = set(), {}
            for f in gt:
                if is_match(f, fields.get(f), gt[f]):
                    ok.add(f)
                else:
                    ng[f] = fields.get(f)
            results[case_id][cond] = {'ok': ok, 'ng': ng, 'doc_angle': res['doc_angle']}
            if verbose:
                print(f"[{case_id}/{cond}] doc_angle={res['doc_angle']} "
                      f"ok={sorted(ok)} ng={ng}", flush=True)
    elapsed = time.time() - t0

    # ---- ケース別スコア表 ----
    print(f"\n評価対象フィールド: total / tax / deposit / change / "
          f"registration_number / store_branch(ケースごとにGTに存在するもののみ)")
    print(f"パイプライン: OnnxReceiptOCR(PP-OCRv6_small) + extract_fields  "
          f"({elapsed:.0f}s)\n")
    header = "| case | GT項目数 | " + " | ".join(conditions) + " |"
    print(header)
    print("|---" * (len(conditions) + 2) + "|")
    for case_id in sorted(results):
        n = len(gts[case_id])
        cells = []
        for cond in conditions:
            r = results[case_id].get(cond)
            cells.append(f"{len(r['ok'])}/{n}" if r else "-")
        print(f"| {case_id} | {n} | " + " | ".join(cells) + " |")

    # ---- 条件別合計・フィールド別内訳 ----
    print("\n| 集計 | " + " | ".join(conditions) + " |")
    print("|---" * (len(conditions) + 1) + "|")
    total_gt = sum(len(g) for g in gts.values())
    row = []
    for cond in conditions:
        okn = sum(len(r['ok']) for c in results.values() if (r := c.get(cond)))
        row.append(f"{okn}/{total_gt} ({okn / total_gt:.0%})")
    print("| 全フィールド | " + " | ".join(row) + " |")
    for f in FIELDS:
        cases_with = [c for c in gts if f in gts[c]]
        if not cases_with:
            continue
        cells = []
        for cond in conditions:
            okn = sum(1 for c in cases_with
                      if (r := results[c].get(cond)) and f in r['ok'])
            cells.append(f"{okn}/{len(cases_with)}")
        print(f"| {f} | " + " | ".join(cells) + " |")

    # ---- 誤りの内訳(条件別) ----
    print("\n--- 不一致の内訳(expected → got, Noneは未検出) ---")
    for cond in conditions:
        lines = []
        for case_id in sorted(results):
            r = results[case_id].get(cond)
            if not r or not r['ng']:
                continue
            det = ", ".join(f"{f}: {gts[case_id][f]} → {got}"
                            for f, got in sorted(r['ng'].items()))
            lines.append(f"  {case_id}: {det}")
        print(f"[{cond}] " + ("全一致" if not lines else ""))
        for line in lines:
            print(line)


if __name__ == '__main__':
    main()
