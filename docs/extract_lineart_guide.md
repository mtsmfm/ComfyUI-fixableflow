# Extract Line Art Node 使用ガイド

## 概要
ComfyUI-LayerDividerに追加された線画抽出ノードです。RGB画像から線画を抽出し、背景を透過させたRGBA画像を生成します。

## ノード種類

### 1. LayerDivider - Extract Line Art
基本的な線画抽出ノード

**入力:**
- `image`: 入力画像（RGB or RGBA）
- `white_threshold`: 白と判定する閾値（0-255、デフォルト: 200）
- `apply_smoothing`: スムージング適用の有無（デフォルト: True）
- `invert_alpha`: アルファチャンネルを反転（デフォルト: False）

**出力:**
- `image`: 背景透過済みのRGBA画像
- `alpha_mask`: アルファチャンネルマスク

### 2. LayerDivider - Extract Line Art Advanced
高度な線画抽出ノード（色付き線画対応）

**入力:**
- `image`: 入力画像
- `white_threshold`: 白と判定する閾値（0-255）
- `apply_smoothing`: スムージング適用
- `preserve_colors`: 線の色を保持（色付き線画用）
- `line_darkness`: 線の濃さ調整（0.0-2.0）
- `edge_detection`: エッジ検出を使用

**出力:**
- `rgba_image`: 背景透過済み画像
- `preview`: チェッカーボード背景付きプレビュー
- `alpha_mask`: アルファマスク

## 使用例

### 基本的な使い方
1. 画像読み込みノードで線画を読み込む
2. `LayerDivider - Extract Line Art`ノードに接続
3. 必要に応じてパラメータを調整
4. 出力された透過画像を保存または後続処理に使用

### ワークフロー例
```
[Load Image] → [Extract Line Art] → [Save Image (RGBA)]
                        ↓
                   [Alpha Mask] → [Mask処理]
```

## パラメータ調整のコツ

- **white_threshold**: 
  - 値を下げる（例: 150）→ より多くの部分を線として認識
  - 値を上げる（例: 230）→ より濃い線のみを抽出

- **apply_smoothing**: 
  - ON: 線画のエッジを滑らかに
  - OFF: シャープなエッジを維持

- **preserve_colors**（Advancedのみ）:
  - 色付き線画の場合はONに
  - 黒い線画にしたい場合はOFF

- **line_darkness**（Advancedのみ）:
  - 1.0: 通常の濃さ
  - <1.0: 線を薄く
  - >1.0: 線を濃く

## 注意事項
- 入力画像は白背景の線画を想定しています
- 複雑な背景がある画像の場合は、事前に背景除去処理を行うことを推奨
- エッジ検出モードはOpenCVが必要です（requirements.txtに含まれています）
