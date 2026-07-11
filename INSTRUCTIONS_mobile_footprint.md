# 指示書: モバイル搭載形の確定 — INT8フルスタック計測・ruriサイズ削減・ORT Mobile互換性

## 目的

レシートOCRパイプライン(PP-OCRv6 small + regex抽出 + ruri-v3-30m フォールバック)を
「スマホに載せられる形」として確定させる。具体的には:
1. INT8フルスタック(OCRもラベラーもINT8)のend-to-end精度・速度・メモリを計測する
2. ruri 30m INT8(現状110MB)のサイズを埋め込みテーブル量子化で60MB以下に削減する
3. ONNX Runtime Mobile での動作互換性を検証する(.ort変換+シングルスレッド計測)

## 環境・リポジトリ

- 作業ディレクトリ: `/Users/answer.kazuya/coding/ocr-poc`(gitリポジトリ、直近コミット c81a057)
- Python: `.venv/bin/python`(3.13)。onnxruntime / sentence-transformers / torch 導入済み
- Apple Silicon Mac。計測はCPUのみ

## 現状(c81a057 時点の実測値 = 本タスクのベースライン)

- パイプライン入口は `pipeline.py` の `ReceiptOCR`(camera_guide → OnnxReceiptOCR →
  regex抽出 → ガード付きruriフォールバック)。現状OCRは FP32 固定
- OCRモデル: `onnx_models/*_int8.onnx` 4本で計11.8MB(INT8時ピークRSS 383MB、OCR単体)
- ラベラー: `onnx_models/ruri-v3-30m/model_int8.onnx`(110MB、INT8 Dynamic)。
  FP32版model.onnxはgitignore(`quantize_ruri.py`で再生成可能)。元モデルは `models/ruri-v3-30m`
- 精度(FP32 OCR + fallback): 実写9枚 66/66、合成20枚 158/158、
  external clean 32/33・light 31/33・medium 23/33・heavy 13/33
- 精度(INT8 OCR、regexのみ・fallbackなし): 実写 66/66、
  external clean 30/33・light 29/33・medium 19/33・heavy 11/33
- 評価コマンド: 実写 `.venv/bin/python eval_quant_real.py int8`、外部 `.venv/bin/python
  eval_external.py [--quantized]`、合成 `.venv/bin/python synth/eval_synth.py [--quantized]`

## タスク

### 1. INT8フルスタックのend-to-end計測

1. `pipeline.py` の `ReceiptOCR.__init__` に `quantized=False` 引数を追加し、
   `OnnxReceiptOCR(quantized=quantized)` に渡す(既定値は現行動作を変えないこと)。
2. 次のマトリクスを計測する(fallbackは常にガード付き、camera_guideは無効化して純粋な抽出性能を測る):
   - 実写9枚: {FP32, INT8} × {regexのみ, +fallback} の4構成
   - external 4段階: INT8 + fallback(FP32+fallbackは計測済み: 32/31/23/13)
   - 合成20枚: INT8 + fallback
3. メモリ・速度のプロファイル(INT8フルスタック、実写9枚で計測):
   - ピークRSS: (a) OCRのみ (b) OCR+ラベラー使用時 — ラベラーの追加RSSを分離して報告
   - ラベラー初期化時間(モデルロード+プロトタイプ73文の埋め込み)と、2枚目以降の1枚あたり処理時間
4. 合格基準: 実写 INT8+fallback で66/66維持、external INT8+fallback が regexのみ(30/29/19/11)
   から悪化しないこと。悪化した場合はどのケースがなぜ落ちたか(OCR差かfallback差か)を切り分けて報告。

### 2. ruri 30m のサイズ削減(110MB → 60MB以下目標)

1. まず `model_int8.onnx` 110MBの内訳を確認する(onnxでinitializerサイズを列挙)。
   語彙~10万×256次元のFP32埋め込みテーブル(~105MB)が支配的のはず。Dynamic量子化は
   MatMul重みしか触らないため、Gatherで引かれる埋め込みテーブルが残っている。
2. 埋め込みテーブルの縮小を2案試す:
   - (a) FP16化(~52MB見込み)。Gather出力をCastでFP32へ戻す
   - (b) INT8化(~26MB見込み)。per-row(トークンごと)スケールのDequantizeLinear
3. 各案の検証(**全て同一手順**):
   - `ruri_experiment/ocr_cache.json` の全行(重複除去で~1222行)について、現行model_int8.onnxとの
     埋め込みcos類似度(mean/min)とラベル一致率を計測
   - `eval_labeler.py` 相当のA/B/C再計測(ONNX版ラベラーで実行できるよう小改修してよい)
   - タスク1のend-to-end(実写9枚+external)を縮小版で再実行
4. 採用基準: ラベル一致率99.5%以上かつend-to-endスコア無劣化。満たす最小サイズの案を採用し、
   `onnx_models/ruri-v3-30m/` に別名(例 `model_int8_emb8.onnx`)で保存。既存ファイルは上書きしない。

### 3. ONNX Runtime Mobile 互換性検証

1. 全5モデル(OCR4本のINT8+ruri採用版)を ORT形式(.ort)へ変換する
   (`python -m onnxruntime.tools.convert_onnx_models_to_ort`)。変換エラーが出たモデルは
   原因(未対応op等)を特定して報告。opset問題なら変換で解消してよい(モデル自体の再量子化は不要)。
2. モバイル相当の悲観値計測: SessionOptions で intra_op_num_threads=1 にして
   実写9枚のend-to-end処理時間(INT8フルスタック+fallback)を計測、通常スレッドとの比を報告。
3. required operators 設定(カスタムビルド用の `required_operators.config`)を生成し、
   ORT Mobileカスタムビルドでのランタイムサイズ感の見積もりに使える材料として報告に含める。
4. 変換した .ort は `onnx_models/ort/` に置く。

## 制約

- **git commit はしない**(コミットは発注側が行う)
- **編集禁止**: `receipt_ocr_paddle.py` / `onnx_receipt_ocr.py` / `camera_guide.py` /
  既存の `onnx_models/*.onnx`(上書き禁止。新規ファイル追加は可)
- 編集可: `pipeline.py`(quantized引数の追加のみ)、`ruri_experiment/` 配下、
  `quantize_ruri.py`(埋め込み量子化の追加)。新規スクリプトは `ruri_experiment/` か
  ルートの新規ファイルに
- `ruri_experiment/ocr_cache.json` は上書きしない
- 閾値・ガード(`onnx_line_labeler.py` の `_FALLBACK_LINE_GUARD`、threshold 0.80)は変更しない
  (プロトタイプ文の最適化は評価データへの過学習リスクがあるため本タスクでは扱わない)
- 追加パッケージが必要なら `.venv` に pip install してよい

## 報告(`ruri_experiment/REPORT_mobile.md` に保存)

1. タスク1のマトリクス表(構成×スイートのスコア、RSS内訳、処理時間)
2. 110MBの内訳と、縮小2案の cos類似度/ラベル一致率/A/B/C/end-to-end/最終サイズの比較表、採用案
3. .ort変換の結果(モデル別の可否と原因)、シングルスレッド計測、required_operators の要点
4. 「スマホ搭載時の最終フットプリント見積もり」まとめ: モデル合計サイズ・ピークRSS・
   1枚あたり処理時間(シングルスレッド悲観値と通常値)
5. うまくいかなかった構成もネガティブ結果として数値付きで残すこと
