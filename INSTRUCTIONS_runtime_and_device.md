# 指示書: ラベラー実行コスト削減 と 実機ベンチマークハーネス

## 目的

前タスク(コミット b2922d0)でモバイル搭載形は確定した(モデル合計47.7MB、INT8フルスタックで
実写66/66)。残る2課題に取り組む:
1. **ラベラーの実行コスト削減** — 追加RSS約300MB(ORTアリーナ+固定長パディングの活性化メモリ)と
   初期化0.7sを削る
2. **実機計測ハーネス** — iOS/Android実機でモデル推論時間を計測できるベンチアプリを、
   実機がなくても検証可能な形で納品する

## 環境・リポジトリ

- 作業ディレクトリ: `/Users/answer.kazuya/coding/ocr-poc`(gitリポジトリ、直近コミット b2922d0)
- Python: `.venv/bin/python`(3.13)
- Apple Silicon Mac

## 現状ベースライン(b2922d0、全て独立プロセスでの実測)

| 構成 | スコア | 速度 | ピークRSS |
|---|---|---|---|
| INT8 OCRのみ | 実写66/66 | 441ms/枚 | 424MB |
| INT8フルスタック(+emb8ラベラー) | 実写66/66 | 568ms/枚 | 728MB |
| 同シングルスレッド | 実写66/66 | 760ms/枚 | 724MB |

- ラベラー: `ruri_experiment/onnx_line_labeler.py` の `OnnxLineLabeler`
  (`onnx_models/ruri-v3-30m/model_int8_emb8.onnx` 35.9MB)。初期化0.7s
  (モデルロード+プロトタイプ73文の埋め込み計算)
- 現実装の特徴: `_tokenize` が **max_length=128固定パディング**、`_encode` が **batch=32固定**。
  レシート行は短いので大半がパディングの無駄計算=活性化メモリと速度の両方に効いているはず

## タスク1: ラベラーの実行コスト削減

目標: ラベラー追加RSSを300MB→**150MB以下**、フルスタック処理時間の短縮、初期化0.7s→0.2s以下。
精度の合格条件(全て必須): 実写9枚66/66(INT8フルスタック)、external INT8+fallbackが
32/31/23/12から悪化しない、1222行ラベル一致率99%以上(比較基準は現行実装。
検証スクリプトの書き方は `ruri_experiment/REPORT_mobile.md` の検証節を参照)。

試すこと(それぞれ効果を分離計測してから組み合わせる):
1. **動的パディング**: まず `ocr_cache.json` 全行のトークン長分布(p50/p95/p99/max)を計測して
   報告に含める。その上でバッチ内最大長へのパディングに変更し、上限はp99程度+余裕に設定
   (上限超は切り詰め。切り詰め発生率も報告)
2. **バッチサイズ感度**: 32/16/8/4 でRSSと速度のトレードオフを計測
3. **SessionOptions**: `enable_cpu_mem_arena=False`、`enable_mem_pattern=False` の効果
   (速度悪化とのトレードオフを計測)
4. **プロトタイプ埋め込みのディスクキャッシュ**: 初回計算した73文の埋め込みを
   `onnx_models/ruri-v3-30m/proto_emb_cache.npz` に保存し、次回以降はロードのみ。
   キャッシュキーにモデルファイルサイズ+プロトタイプ文数を入れ、変わったら再計算
5. (余力があれば)OCR側もSessionOptionsの同項目を試す(424MBの削減余地確認のみ。
   `onnx_receipt_ocr.py` は編集禁止なので計測はモンキーパッチで行い、結果報告のみ)

計測の作法(前回指摘済みの再発防止):
- RSS計測は必ず**構成ごとに独立プロセス**(`resource.getrusage`)。1プロセス内での差分報告は不可
- スコアは必ず9枚全部で集計(66/66が満点)

## タスク2: 実機ベンチマークハーネス

方針: 前後処理(検出後処理・CTC・行再構成・regex)の ネイティブ移植は本タスクの範囲外。
**モデル推論だけを実機で正確に計測する**ため、Python側で実データから中間テンソルをダンプして
アプリに同梱する。

1. **入力テンソルのダンプ**: `mobile/testdata/` に、実レシート(receipt.jpg と receipt9.jpg)から
   次をnpy(またはバイナリ+形状メタJSON)で保存するスクリプトを書く:
   - doc_ori入力(1,3,224,224)、det入力(実寸にリサイズ済み)、
     rec入力(実際の切り出しcrop数枚分、(1,3,48,W))、textline_ori入力、
     ラベラー入力(input_ids/attention_mask、実際のOCR行から)
2. **Androidベンチアプリ** `mobile/android/`:
   - Kotlin + `com.microsoft.onnxruntime:onnxruntime-android`(Maven)。最小構成のGradleプロジェクト
   - assetsに .ort 5本(`onnx_models/ort/`。無ければ変換コマンドはREPORT_mobile.md参照)と
     テストテンソルを同梱
   - 起動→全モデルをロード→各テンソルでN=10回推論→モデル別 平均/最悪ms と
     ネイティブヒープ使用量(`Debug.getNativeHeapAllocatedSize`)を画面とlogcatに出力
   - レシート1枚相当の合成値(doc_ori 1回+det 1回+rec×crop数+textline×crop数+ラベラー1回)も表示
3. **iOSハーネス雛形** `mobile/ios/`: 同等のSwift実装(onnxruntime-objc / SPM)。
   Xcodeが無い環境ではビルド確認まで求めない(コードとREADMEを納品物とする)
4. **ツールチェーン確認**: Android SDK/gradle/adb、Xcodeの有無をまず確認して報告。
   あればエミュレータ/シミュレータでスモーク実行(数値は参考値と明記)。
   無ければ「ビルド可能なプロジェクト+実行手順README」までを納品物とし、
   その場合もGradleの構成が妥当かは `gradle assembleDebug` 相当の静的確認でベストエフォート
5. `mobile/.gitignore` を作り、build成果物・ローカルSDK設定(local.properties等)を除外する

## 制約

- **git commit はしない**
- **編集禁止**: `receipt_ocr_paddle.py` / `onnx_receipt_ocr.py` / `camera_guide.py` /
  `pipeline.py` / 既存 `onnx_models/*.onnx`(上書き禁止) / `ruri_experiment/ocr_cache.json` /
  ラベラーの閾値0.80と `_FALLBACK_LINE_GUARD` / プロトタイプ文
- 編集可: `ruri_experiment/onnx_line_labeler.py`(トークナイズ・バッチ・SessionOptions・
  プロトタイプキャッシュ)、新規 `mobile/` 配下、新規計測スクリプト
- **納品物の合格条件(重要)**: 変更後のラベラー・生成したモデル/キャッシュは、必ず
  「実際にロードして実推論し、上記の精度合格条件を通す」こと。ロード確認をしていない
  成果物を「採用」「成功」と報告しない。動かない中間生成物は削除するか、
  明確に「動作しない実験痕」と報告に書く
- 追加パッケージは `.venv` に。Android/iOS依存はプロジェクト内で完結させる

## 報告(`ruri_experiment/REPORT_runtime.md` に保存)

1. タスク1: 施策別の効果分離表(RSS/速度/精度)と採用構成、トークン長分布、
   最終のフルスタック実測(独立プロセス、実写66/66確認込み)
2. タスク2: ツールチェーン確認結果、プロジェクト構成、(実行できた場合)モデル別推論時間、
   実機で実行する手順(ユーザーがスマホを接続して叩くコマンドまで具体的に)
3. ネガティブ結果も数値付きで
