# ocr-mb

日本語のレシート・領収書に特化した、**スマホ単体で完結するOCR**のPoC(概念実証)。
サーバーに画像を送らず、外部APIにも依存しない。商用利用を見据えて、精度・モデルサイズ・
実行速度のバランスを検証している。

> **現状**: モデル・抽出ロジック・実機での実行可能性の検証は完了。写真を撮って結果が返る
> 実アプリ(PWA予定)はこれから実装するフェーズ。

## できていること

| 項目 | 結果 |
|---|---|
| OCRモデル | PP-OCRv6 small(検出+認識+文書/行向き分類、4モデル) |
| フィールド抽出フォールバック | ruri-v3-30m(名古屋大、埋め込みベースの行ラベリング) |
| モデル合計サイズ | **47.7 MB**(OCR 11.8MB + ruri 35.9MB、いずれもINT8) |
| 精度(実写レシート、ローカル評価) | 66/66 |
| 精度(自作合成レシート20枚) | 158/158 |
| 精度(外部ベンチマーク K10124、劣化画像込み) | clean 32/33・light 31/33・medium 23/33・heavy 13/33 |
| 実機推論(Android実機、レシート1枚相当) | 547〜641 ms |
| ピークメモリ(デスクトップCPU、フルスタック) | 約554 MB |

すべて実測値。詳しい経緯は [`note-article-draft.md`](note-article-draft.md) にまとめてある。

## パイプライン

```
撮影画像
  → camera_guide.py    画質チェック(暗すぎ/解像度不足はここで再撮影を促す)
  → onnx_receipt_ocr.py 文書向き補正 → テキスト検出(DB) → 行向き補正 → 文字認識(CTC)
  → receipt_ocr_paddle.py  正規表現によるフィールド抽出(店名・日付・金額・登録番号 等)
  → ruri_experiment/onnx_line_labeler.py  正規表現が未検出のときだけ埋め込みモデルで補完
  → フィールドdict(store_brand, total, tax, deposit, change, date, time, ...)
```

PaddlePaddleには依存しない。すべてONNX Runtime + OpenCV + pyclipperのみで動く自作パイプライン。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.11+ を想定(開発は3.13で実施)。

## 使い方

```bash
# 自分のレシート写真で試す(手持ちの画像を receipt.jpg として置く)
.venv/bin/python pipeline.py receipt.jpg

# 合成データ20枚で評価(公開データのみで再現可能)
.venv/bin/python synth/eval_synth.py

# 外部ベンチマーク(K10124, CC BY 4.0)で評価(公開データのみで再現可能)
.venv/bin/python eval_external.py
```

### 実写レシートで評価したい場合

個人の購買データ(店名・日時・金額・インボイス登録番号)が写り込むため、実写レシートの
正解値はこのリポジトリに含めていない。自分のレシートで評価する場合は、`receipt.jpg` /
`receipt2.jpg` ... をリポジトリ直下に置き、`local_ground_truth.py`(`.gitignore`対象)を
下記の形式で作成する。

```python
GROUND_TRUTHS = {
    'receipt.jpg': {
        'store_brand': 'LAWSON', 'store_branch': '○○店',
        'registration_number': 'T0000000000000',
        'date': '2026-01-01', 'time': '12:00',
        'total': 500, 'tax': 45, 'deposit': 1000, 'change': 500,
    },
}
GROUND_TRUTH = GROUND_TRUTHS['receipt.jpg']  # receipt_ocr_paddle.py 単体デモ用
```

`benchmark_models.py` / `eval_quant_real.py` / `onnx_receipt_ocr.py` はこのファイルが
無ければ実写評価を自動的にスキップする(合成・外部データの評価には影響しない)。

## リポジトリ構成

```
receipt_ocr_paddle.py   フィールド抽出の本体(正規表現・ブランド辞書・整合性補正)
onnx_receipt_ocr.py     OCR推論パイプライン(検出・認識・向き分類・条件付き前処理)
camera_guide.py         撮影時の画質チェック
pipeline.py             上記3つを繋いだ入口(camera_guide → OCR → regex → fallback)
onnx_models/            変換済みONNXモデル(FP32/INT8)
ruri_experiment/        埋め込みモデル(ruri-v3-30m)によるフォールバック抽出の実装・検証
synth/                  架空店舗の合成レシート生成器・評価スクリプト
external_data/          外部ベンチマークデータ(K10124, CC BY 4.0)
degrade.py              スマホ撮影の劣化(手ブレ・影・感熱紙の掠れ等)を合成するツール
mobile/                 モデル推論の実機ベンチアプリ(Android/iOS)・実機用テストデータ
quantize_*.py           INT8量子化スクリプト(モデルの再生成用、推論には不要)
```

## ライセンス

このリポジトリのコードは [Apache License 2.0](LICENSE) の下で公開している。

同梱・依存しているモデル・データのライセンス:

- PP-OCRv6(検出・認識・向き分類): Apache-2.0
- ruri-v3-30m(名古屋大学): Apache-2.0
- 外部評価データ K10124: CC BY 4.0(合成データ、実在店舗・個人を含まない)

## 次にやること

- PWAとしての実装(ONNX Runtime WebのWASMバックエンドでブラウザ内完結を予定)
- iOS実機での計測(現状Android実機のみ検証済み)
- 支払手段の抽出、実写サンプルの追加
