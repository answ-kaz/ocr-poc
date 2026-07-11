#!/usr/bin/env python3
"""
ruri-v3-30m ONNX版行ラベラー — regexフォールバック用。

sentence-transformersに依存せずONNX Runtime + Tokenizerのみで動作する。
使い方:
  from ruri_experiment.onnx_line_labeler import OnnxLineLabeler
  labeler = OnnxLineLabeler()  # 初回ロード時にプロトタイプ埋め込みを事前計算
  labels = labeler.label_lines(['合計 ¥494', 'お預り ¥1,000', ...])
  fields = labeler.extract_from_labels(lines, labels)

モデル: onnx_models/ruri-v3-30m/model_int8_emb8.onnx
       (INT8 dynamic + 埋め込みテーブルINT8 per-row、35.9MB)
"""
import json
import os
import re
import unicodedata

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_MODEL = os.path.join(ROOT, 'onnx_models', 'ruri-v3-30m', 'model_int8_emb8.onnx')
TOKENIZER_PATH = os.path.join(ROOT, 'onnx_models', 'ruri-v3-30m', 'tokenizer.json')

# プロトタイプ・ディストラクタ(共通)
from ruri_experiment.line_labeler import (
    PROTOTYPES, DISTRACTORS, DEFAULT_THRESHOLD,
    normalize_line, extract_amount, extract_date, extract_time,
    extract_branch, extract_regnum, _RATE_RE
)


class OnnxLineLabeler:
    def __init__(self, model_path=ONNX_MODEL, threshold=DEFAULT_THRESHOLD, dims=None,
                 batch_size=32, max_length=40):
        self.threshold = threshold
        self.dims = dims
        self.batch_size = batch_size
        self.max_length = max_length

        # ONNXセッション(メモリ最適化)
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        opt.enable_cpu_mem_arena = False
        opt.enable_mem_pattern = False
        self.sess = ort.InferenceSession(model_path, opt,
                                         providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.sess.get_inputs()]

        # トークナイザ
        with open(TOKENIZER_PATH, encoding='utf-8') as f:
            tok_config = json.load(f)
        self._build_tokenizer(tok_config)

        # プロトタイプ埋め込みを事前計算(キャッシュ確認)
        self.proto_texts = []
        self.proto_labels = []
        for label, sents in PROTOTYPES.items():
            for s in sents:
                self.proto_texts.append(s)
                self.proto_labels.append(label)
        for s in DISTRACTORS:
            self.proto_texts.append(s)
            self.proto_labels.append('other')
        self.proto_emb = self._load_or_compute_proto_emb(model_path)
        if self.dims is not None:
            self.proto_emb = self.proto_emb[:, :self.dims]
            norms = np.linalg.norm(self.proto_emb, axis=1, keepdims=True)
            self.proto_emb = self.proto_emb / np.maximum(norms, 1e-12)

    def _build_tokenizer(self, config):
        """tokenizer.jsonから簡易トークナイザを構築(SentencePiece BPE)"""
        import sentencepiece as spm
        sp_model = os.path.join(os.path.dirname(TOKENIZER_PATH), 'tokenizer.model')
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)

    def _load_or_compute_proto_emb(self, model_path):
        """プロトタイプ埋め込みをキャッシュからロード、なければ計算して保存。

        キャッシュは実際にロードしたモデルのパスに紐づける(別モデルの
        インスタンスが他モデルのキャッシュを拾わないように)。
        キー: モデルサイズ + プロトタイプ文数 + トークン長上限
        """
        cache_path = model_path + '.proto_cache.npz'
        cache_key = (f'{os.path.getsize(model_path)}_'
                     f'{len(self.proto_texts)}_{self.max_length}')

        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            if 'cache_key' in data and str(data['cache_key']) == cache_key:
                return data['proto_emb']

        emb = self._encode(self.proto_texts)
        np.savez(cache_path, proto_emb=emb, cache_key=cache_key)
        return emb

    def _tokenize(self, texts, max_length=None):
        """テキストリスト→(input_ids, attention_mask)のnumpy配列。
        動的パディング: バッチ内最大長にパディング、上限はmax_length。"""
        if max_length is None:
            max_length = self.max_length

        # まず全テキストをトークン化
        all_ids = []
        for text in texts:
            normalized = normalize_line(text)
            ids = self.sp.encode(normalized, out_type=int)
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
            all_ids.append(ids)

        # バッチ内最大長を計算(上限付き)
        batch_max = max(len(ids) for ids in all_ids)
        batch_max = min(batch_max, max_length)

        # パディング
        batch_ids, batch_mask = [], []
        for ids in all_ids:
            if len(ids) < batch_max:
                pad_len = batch_max - len(ids)
                ids = ids + [0] * pad_len
                mask = [1] * (len(ids) - pad_len) + [0] * pad_len
            else:
                ids = ids[:batch_max]
                mask = [1] * batch_max
            batch_ids.append(ids)
            batch_mask.append(mask)

        return (np.array(batch_ids, dtype=np.int64),
                np.array(batch_mask, dtype=np.int64))

    def _pool(self, output, attention_mask):
        """トークン埋め込みの平均プーリング + L2正規化"""
        if output.ndim == 3:
            output = output[0]
        mask = attention_mask.astype(bool)
        if mask.any():
            emb = output[mask].mean(axis=0)
        else:
            emb = output.mean(axis=0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def _encode(self, texts):
        """テキストリスト→埋め込み配列 (n, dim)"""
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            ids, mask = self._tokenize(batch)
            feeds = {'input_ids': ids, 'attention_mask': mask}
            out = self.sess.run(None, feeds)
            for j in range(len(batch)):
                emb = self._pool(out[0][j], mask[j])
                all_embs.append(emb)
        return np.array(all_embs)

    def encode_lines(self, lines):
        return self._encode(lines)

    def label_from_embeddings(self, line_emb, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        emb = line_emb
        proto = self.proto_emb
        if self.dims is not None:
            emb = emb[:, :self.dims]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norms, 1e-12)
            proto = self.proto_emb
        sims = emb @ proto.T
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

    def extract_from_labels(self, lines, labels):
        """label_lines()の出力からフィールドdictを構築(共通ロジック)"""
        by_field = {}
        for line, (label, score) in zip(lines, labels):
            if label != 'other':
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


FALLBACK_FIELDS = ('total', 'deposit', 'change', 'tax', 'date', 'time',
                   'store_branch', 'registration_number')

# fallback補完時の語彙ガード: deposit/changeはレシート上の語彙が閉じているため、
# 採用する行にコア文字を要求する。埋め込みラベラーは商品行や支払手段行を
# deposit/changeと誤ラベルすることがあり(実測: 「おにぎりソーセージマヨ 322軽」
# →change=322、「SCギフト 1 ¥1,800」→deposit=1800)、これを遮断する。
# total/taxは表記ゆれへの汎化がラベラーの価値なのでガードしない
_FALLBACK_LINE_GUARD = {
    'deposit': re.compile(r'預|現\s*金'),
    'change': re.compile(r'[釣鈎釘]|つり'),
}

_shared_labeler = None


def get_labeler():
    """プロセス内で共有するラベラー(モデルロードとプロトタイプ埋め込みは初回のみ)"""
    global _shared_labeler
    if _shared_labeler is None:
        _shared_labeler = OnnxLineLabeler()
    return _shared_labeler


def apply_line_guard(lines, labels):
    """deposit/changeラベルのうちコア文字を含まない行を'other'へ落とす"""
    guarded = []
    for line, (label, score) in zip(lines, labels):
        guard = _FALLBACK_LINE_GUARD.get(label)
        if guard is not None and not guard.search(line):
            label = 'other'
        guarded.append((label, score))
    return guarded


def extract_fields_with_fallback(text, extra_brands=None, labeler=None):
    """regex extract_fieldsの結果に、未検出フィールドがあればラベラーで補完。

    1. regexで抽出
    2. 未検出フィールドがあればラベラーで再抽出(deposit/changeは語彙ガード付き)
    3. regex結果を優先し、ラベラーは補完のみ(既存正解を壊さない)
    """
    from receipt_ocr_paddle import extract_fields

    regex_fields = extract_fields(text, extra_brands=extra_brands)
    if all(f in regex_fields for f in FALLBACK_FIELDS):
        return regex_fields

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return regex_fields

    labeler = labeler or get_labeler()
    labels = apply_line_guard(lines, labeler.label_lines(lines))
    labeler_fields = labeler.extract_from_labels(lines, labels)

    merged = dict(regex_fields)
    for field in FALLBACK_FIELDS:
        if field not in merged and field in labeler_fields:
            merged[field] = labeler_fields[field]
    return merged
