# External Data Manifest — 日本語レシートOCR 評価用公開データセット調査

調査日: 2026-07-10 / 調査方法: HF API + README + 実データ精査(画像目視含む)
前提: 本プロジェクトは**商用利用予定**。ライセンス確認を最優先とした。

## サマリ表

| データセット | 出典 | ライセンス | 商用可否 | 枚数 | 実写/合成 | アノテーション形式 | DL状況 |
|---|---|---|---|---|---|---|---|
| K10124/japan-ocr-mini-benchmark (**alpha10のみ**) | https://huggingface.co/datasets/K10124/japan-ocr-mini-benchmark | CC BY 4.0(alpha10ペイロードに限り明示確認あり) | **可**(表示義務あり) | 10枚 | 合成(クリーンなレンダリング) | 構造化JSON(店舗情報・品目行・税サマリ・合計・支払)+ manifest CSV/JSON。**全文テキストGT・bboxなし** | 済 → `K10124__japan-ocr-mini-benchmark/alpha10/` |
| systemk-ai/receipt-ocr-ja | https://huggingface.co/datasets/systemk-ai/receipt-ocr-ja | **不明(gatedで未確認)** | 不明 | 475例 | 不明(スキーマはminato-ryanと同一 → 同系の生成画像と推定) | 構造化JSON(issuer/items/summary)+ caption + image | **未DL: gated(manual承認制)。HF未ログインのためアクセス不可** |
| minato-ryan/receipt-ocr-ja-example | https://huggingface.co/datasets/minato-ryan/receipt-ocr-ja-example | **未指定(license: null、README本文にも記載なし)** | **不可扱い**(無指定=著作権留保。さらに実在ブランド名を多数含む) | 15例(うち日本語9、英語3、中国語3) | 合成(画像生成AIによる写実的レシート写真、768x1376固定) | 構造化JSON(issuer/items/summary)+ 生成用caption。全文GT・bboxなし | **external_dataには未保存**(精査はscratchpadで実施) |
| Aulvem/japanese-invoice-receipt-extraction-eval | https://huggingface.co/datasets/Aulvem/japanese-invoice-receipt-extraction-eval | CC BY-NC 4.0 | **不可(非商用)** | invoice 20 + receipt 10(サンプルJSONL) | — | **画像なし**。`document_text`(テキスト)→ `expected_output`(JSON)のテキスト抽出ベンチ | 未DL(注記のみ) |

## 各データセット詳細

### 1. K10124/japan-ocr-mini-benchmark — alpha10 ✅ 推奨(唯一の商用可)

- 保存先: `external_data/K10124__japan-ocr-mini-benchmark/alpha10/`
- ライセンス: CC BY 4.0。`alpha10/LICENSE_NOTICE.md` と `ALPHA10_LICENSE_FINAL_CONFIRMATION.md` に「Alpha10 approved payloadに限り確認済み」と明記。**リポジトリ内の他バージョン(v0.2.0等)は対象外なのでダウンロードしていない。**
- 内容: 合成日本語レシート10枚(PNG, RGB)。解像度は 350x424 〜 960x1660(レシートらしい縦長)。
- 業種バリエーション: 和菓子店・スーパー(生活雑貨)・そば屋・居酒屋・タクシー・温泉施設など。8%/10%混在税率、現金/クレジット/電子マネー/コード決済と支払手段も多様。
- アノテーション: `source_json/` に品目名・数量・単価・金額・税率・小計・合計・預り金・釣銭・税サマリ(端数処理ポリシー付き)。`metadata/` にSHA-256・safety情報。`alpha10_manifest.{csv,json}` が索引。
- 品質所感(目視2枚): レイアウト・フォント・罫線とも実際の日本語レシートに忠実で読みやすい。「クリーン」画像のみで、ノイズ・傾き・影・ブレなし。数値の整合(小計=品目合計、内税計算)も確認できた。
- 制約: (a) 全文テキストGTとbboxがないため、検出(det)評価や行単位のCER計測にはJSONから期待文字列を再構成する必要がある。(b) クリーン画像のみなのでスマホ実写条件(影・歪み)の評価にはならない → 自前で撮影劣化を合成する余地あり。(c) 10枚と少ない。

### 2. systemk-ai/receipt-ocr-ja — gatedで取得不可

- 475例・68MB parquet。スキーマ(language/aspect_ratio/visual_style/data/caption/image)がminato-ryanと完全一致しており、同じ生成パイプラインの大規模版と推定。
- `gated: manual`(リポジトリ作者の手動承認制)。この環境はHF未ログインのため取得不可。
- 使いたい場合: HFアカウントでアクセス申請 → 承認後にライセンス条件を確認、が必要。minato-ryan同様に実在ブランド名を含む可能性が高い点に注意。

### 3. minato-ryan/receipt-ocr-ja-example — 商用不可扱い(未保存)

- ライセンス表記が一切なし(HF API `license: null`、READMEはdataset_info YAMLのみ)。無指定 = 権利留保がデフォルトで、商用利用の根拠がない。
- さらに内容面のリスク: issuer名に **イオンスタイル幕張新都心、ファミリーマート、(株)セブン-イレブン・ジャパン、トップバリュ(商品名)、海底捞、蜜雪冰城、全聚德** など実在ブランドが多数登場する。商用プロダクトの評価データとしては商標・パブリシティ面でも不適。
- 品質自体は高い: 768x1376の画像生成AI製「レシートを撮影した写真」風(背景・照明・紙のカール等あり)で、構造化GT(issuer/items/taxRate/summary)と画像内容の整合を1例で全項目一致確認。全文GT・bboxはなし。15例中日本語は9例。
- 判断: 精査はscratchpad上でのみ実施し、external_dataには保存していない。

### 4. Aulvem/japanese-invoice-receipt-extraction-eval — 非商用 + 画像なし

- CC BY-NC 4.0(非商用)なので商用プロジェクトでは使用不可。フル版(2,000〜5,000件)は商用ライセンスで別売(aulvem.com / Gumroad)。
- そもそも**画像を含まない**「テキスト→JSON構造化抽出」ベンチ(和暦・軽減税率・源泉徴収などの難所入り)。OCR画像評価には無関係で、使うとすればOCR後段の構造化パイプライン評価のみ。
- 本プロジェクト(OCR画像評価)の目的には合致しないため未ダウンロード。

## 結論

- **今すぐ評価に使えるのは K10124 alpha10 の10枚のみ**(CC BY 4.0、要クレジット表記)。合成クリーン画像なので「認識精度の下限確認・回帰テスト」用途に向く。スマホ実写条件の評価には、alpha10に透視変換・影・ノイズを合成して疑似実写化するか、実写レシートの追加撮影が必要。
- systemk-ai(475例)は量的に魅力だが、gated承認とライセンス確認が通るまで保留。
- minato-ryan・Aulvemは商用要件を満たさないため不採用。
