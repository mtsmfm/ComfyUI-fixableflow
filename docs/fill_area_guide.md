# Fill Area Node 使用ガイド

## 概要
ComfyUI-LayerDividerに追加された塗り領域均一化ノードです。線画の輪郭を検出し、各領域を単色で塗りつぶすことで、クリーンな塗り画像を生成します。

## ノード種類

### 1. LayerDivider - Fill Area
基本的な塗り領域均一化ノード

**入力:**
- `binary_image`: 輪郭画像（線画）
- `fill_image`: 塗り画像
- `min_area_pixels`: 小領域と判定するピクセル数（10-10000、デフォルト: 1000）
- `similarity_threshold`: 色の類似度閾値（1-100、デフォルト: 10）

**出力:**
- `filled_image`: 均一化された塗り画像
- `preview`: 処理前後の比較画像

### 2. LayerDivider - Fill Area Advanced
高度な塗り領域均一化ノード

**入力:**
- `binary_image`: 輪郭画像（線画）
- `fill_image`: 塗り画像
- `min_area_pixels`: 小領域判定の閾値
- `similarity_threshold`: 色類似度の閾値
- `edge_enhancement`: エッジ強調の有無
- `preserve_alpha`: アルファチャンネルを保持
- `output_mode`: 出力モード選択
  - `filled`: 均一化された画像
  - `contours`: 輪郭の可視化
  - `labels`: ラベルの可視化
  - `comparison`: 処理前後の比較

**出力:**
- `filled_image`: 均一化された画像
- `visualization`: 選択されたモードの可視化
- `original`: 元の塗り画像
- `contour_mask`: 輪郭マスク

## 使用例

### 基本的なワークフロー
```
[Load Image (線画)] → [Fill Area] → [Save Image]
        ↓                 ↑
[Load Image (塗り画像)] →┘
```

### 高度なワークフロー例
```
[Load Image (線画)] → [Extract Line Art] → [Fill Area Advanced] → [Save Image]
                                    ↑            ↓
[Load Image (ラフな塗り)] →→→→→→→→→→→┘        [Visualization]
```

## パラメータ調整のコツ

### min_area_pixels（小領域の閾値）
- **小さい値（10-500）**: 細かい部分も保持
- **中程度（500-2000）**: 標準的な設定
- **大きい値（2000-10000）**: 小さな領域を積極的に統合

### similarity_threshold（色の類似度）
- **低い値（1-5）**: 厳密な色の区別
- **中程度（5-20）**: 標準的な設定
- **高い値（20-100）**: 類似色を積極的に統合

### output_mode（Advancedのみ）
- **filled**: 最終的な塗り画像
- **contours**: 輪郭線のみを表示（デバッグ用）
- **labels**: 各領域を異なる色で表示（領域確認用）
- **comparison**: 処理前後を並べて表示

## 処理の流れ

1. **輪郭検出**: 線画から閉じた領域を検出
2. **色の抽出**: 各領域から最頻出色を取得
3. **小領域の統合**: 指定サイズ以下の領域を近い色の領域に統合
4. **類似色の統合**: 色空間で近い色の領域を統合
5. **塗りつぶし**: 各領域を単一色で塗りつぶし

## 活用シーン

- **イラストのクリーンアップ**: ラフな塗りを均一化
- **アニメーション準備**: セル画風の塗り分け
- **マスク生成**: 領域ごとのマスク作成
- **色の簡略化**: 複雑な塗りをシンプルに

## 注意事項

- 線画は閉じた輪郭である必要があります
- 開いた線の場合、領域が正しく検出されない可能性があります
- 処理には以下のライブラリが必要です：
  - scipy
  - scikit-image
  - opencv-python

## トラブルシューティング

### 領域が検出されない場合
- 線画の輪郭が閉じているか確認
- エッジ強調オプションを試す（Advanced）

### 色が統合されすぎる場合
- similarity_thresholdを下げる
- min_area_pixelsを下げる

### 小さい領域が残る場合
- min_area_pixelsを上げる
- similarity_thresholdを上げる
