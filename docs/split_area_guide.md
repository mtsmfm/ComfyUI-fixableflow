# Split Area Node 使用ガイド

## 概要
ComfyUI-LayerDividerに追加された領域分割ノードです。線画の線を太くして領域を分割し、各領域を異なる色で塗り分けることで、領域マスクや色分けマップを生成します。

## ノード種類

### 1. LayerDivider - Split Area
基本的な領域分割ノード

**入力:**
- `lineart_image`: 線画画像
- `thickness`: 線の太さ（1-10、デフォルト: 1）
- `threshold`: 二値化の閾値（0-255、デフォルト: 128）
- `fill_image`（オプション）: 塗り画像
- `use_fill_colors`（オプション）: 塗り画像の色を使用
- `random_seed`（オプション）: ランダムシード（-1でランダム）

**出力:**
- `split_image`: 色分けされた画像
- `binary_image`: 二値化画像
- `region_mask`: 領域マスク

### 2. LayerDivider - Split Area Advanced
高度な領域分割ノード

**入力:**
- `lineart_image`: 線画画像
- `thickness`: 線の太さ（1-20）
- `threshold`: 二値化閾値
- `dilation_iterations`: 膨張処理の回数（1-5）
- `color_mode`: 色付けモード
  - `random`: ランダム色
  - `gradient`: グラデーション
  - `pastel`: パステルカラー
  - `vivid`: ビビッドカラー
  - `from_fill`: 塗り画像から色を取得
- `output_mode`: 出力モード
  - `colored`: 色分けされた画像
  - `regions`: 領域の可視化
  - `overlay`: オーバーレイ表示
  - `comparison`: 比較表示
- `fill_image`（オプション）: 塗り画像
- `random_seed`（オプション）: ランダムシード
- `preserve_lines`（オプション）: 元の線を保持
- `line_color`（オプション）: 線の色（R,G,B形式）

**出力:**
- `split_image`: 色分けされた画像
- `visualization`: 可視化画像
- `binary_image`: 二値化画像
- `region_mask`: 領域マスク
- `num_regions`: 検出された領域数

## 使用例

### 基本的なワークフロー
```
[Load Image (線画)] → [Split Area] → [Save Image]
                           ↓
                    [Region Mask] → [Mask処理]
```

### Fill Areaとの連携ワークフロー
```
[線画] → [Fill Area] → [Split Area Advanced] → [色分け画像]
    ↓                         ↑
[ラフ塗り] →→→→→→→→→→→→→→→→→┘
                              ↓
                        [領域数の確認]
```

### 完全なパイプライン例
```
[元画像] → [Extract Line Art] → [Split Area] → [領域マスク]
                ↓                     ↓
         [Fill Area] ←→→→→→→ [色分けマップ]
                ↓
         [最終的な塗り画像]
```

## パラメータ調整のコツ

### thickness（線の太さ）
- **1-3**: 細い線画向け、詳細な領域分割
- **4-7**: 標準的な設定
- **8-20**: 太い線画向け、大まかな領域分割

### threshold（二値化閾値）
- **低い値（0-100）**: より多くの線を検出
- **中程度（100-150）**: 標準的な設定
- **高い値（150-255）**: 濃い線のみを検出

### dilation_iterations（膨張処理）
- **1**: 最小限の処理
- **2-3**: 線の隙間を埋める
- **4-5**: 積極的に線を太くする

### color_mode（色付けモード）
- **random**: 完全にランダムな色
- **gradient**: 虹色のグラデーション
- **pastel**: 優しいパステル調
- **vivid**: 鮮やかな原色系
- **from_fill**: 塗り画像から色を取得

## 処理の流れ

1. **線の太さ調整**: 指定された太さに線を膨張
2. **二値化処理**: 閾値で線画を二値化
3. **領域検出**: 閉じた領域を検出
4. **色の割り当て**: 各領域に色を割り当て
5. **可視化**: 選択されたモードで出力

## 活用シーン

- **領域マスクの生成**: セグメンテーション用マスク作成
- **色分けマップ**: アニメーション制作の参照用
- **塗り分けガイド**: イラスト制作の下塗り準備
- **領域数の確認**: 画像の複雑さを数値化
- **スタイル転写**: 領域ごとに異なるスタイルを適用

## トラブルシューティング

### 領域が分割されない場合
- thicknessを増やす
- thresholdを調整する
- dilation_iterationsを増やす（Advanced）

### 領域が結合してしまう場合
- thicknessを減らす
- thresholdを上げる
- 線画の隙間を確認

### 色が期待通りにならない場合
- random_seedを固定値に設定
- color_modeを変更
- fill_imageを提供（from_fillモード）

## 他のノードとの連携

### Extract Line Art → Split Area
線画抽出後に領域分割することで、クリーンな領域マップを生成

### Fill Area → Split Area
均一化された塗りを基に、より正確な領域分割を実現

### Split Area → その他の処理
生成された領域マスクを使って、選択的な画像処理が可能

## 注意事項

- 線画は閉じた輪郭である必要があります
- 線の隙間が大きい場合は、thicknessやdilation_iterationsで調整
- 領域数が多すぎる場合はメモリ使用量に注意
- color_mode="from_fill"を使用する場合は、fill_imageが必須
