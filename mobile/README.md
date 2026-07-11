# モバイルベンチマークハーネス

## 概要

レシートOCRパイプライン(PP-OCRv6_small + ruri-v3-30m)のモデル推論時間を
iOS/Android実機で計測するためのベンチアプリ。

## ディレクトリ構成

```
mobile/
├── dump_testdata.py        # テストデータ生成スクリプト
├── testdata/               # ダンプされたテストテンソル(.bin + .json)
├── android/                # Androidベンチアプリ
│   ├── build.gradle
│   ├── settings.gradle
│   └── app/
│       ├── build.gradle
│       └── src/main/
│           ├── AndroidManifest.xml
│           ├── java/.../MainActivity.kt
│           └── assets/    # .ortモデルを配置
└── ios/                    # iOSベンチアプリ
    ├── Package.swift
    └── OCRBench/main.swift
```

## 手順

### 1. テストデータの生成

`mobile/dump_testdata.py` はリポジトリ直下の `receipt.jpg` / `receipt9.jpg` からテンソルを生成する。
これらは個人の実写レシート(購買データ)のため公開リポジトリには含めていない。
自分の実写レシートを `receipt.jpg` として直下に置く(`local_ground_truth.py` は任意)か、
`dump_receipt()` の呼び出し先を手持ちの画像パスに書き換えて実行する。

```bash
cd /path/to/ocr-poc
.venv/bin/python mobile/dump_testdata.py
```

### 2. モデル・テストデータのコピー

```bash
cp onnx_models/ort/*.ort mobile/android/app/src/main/assets/
cp mobile/testdata/* mobile/android/app/src/main/assets/
# iOSはXcodeプロジェクトのリソースに .ort と testdata/* を追加する
```

.ort が無い場合の再生成:
```bash
.venv/bin/python -m onnxruntime.tools.convert_onnx_models_to_ort onnx_models/PP-OCRv6_small_det_int8.onnx --output_dir onnx_models/ort
# (rec/doc_ori/textline_ori/ruri-v3-30m/model_int8_emb8 も同様)
```

### 3. Android版

#### 前提条件
- Android Studio (SDK 34)
- Kotlin対応

#### ビルド・実行
Gradleラッパーは未同梱のため、初回はAndroid Studioでmobile/androidを開くか
`gradle wrapper`(要Gradle 8.x)で生成する。

```bash
cd mobile/android
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.ocrbench/.MainActivity
adb logcat -s OCRBench
```

### 4. iOS版

#### 前提条件
- Xcode 15+
- Swift 5.9+

#### ビルド・実行
```bash
cd mobile/ios
swift build
swift run
```

またはXcodeで `Package.swift` を開き、実機で実行。

## 出力例

```
=== receipt ===
  det: avg=45.2ms max=52.1ms
  rec: avg=12.3ms max=14.8ms (×10 crops)
  textline_ori: avg=8.1ms max=9.5ms (×10 crops)
  labeler: avg=15.6ms max=18.2ms

=== Memory ===
Native heap: 85MB
```

## 合成値(レシート1枚相当)

```
doc_ori: 1回
det: 1回
rec: crop数回 (通常5-15回)
textline_ori: crop数回
labeler: 1回

例: 10 cropの場合
= doc_ori + det + rec×10 + textline×10 + labeler
= 1 + 1 + 10 + 10 + 1 = 23回推論
```

## 注意事項

- テストデータはリトルエンディアンの生バイト列(.bin)+shape/dtypeメタ(.json)
- iOS版はビルド未検証の雛形(この環境にXcodeが無いため)。API差分の調整が必要な可能性あり
- 合成値は実際のOCRパイプラインの前後処理を含まない
- 各数値は参考値(実機の負荷状態等で変動)
