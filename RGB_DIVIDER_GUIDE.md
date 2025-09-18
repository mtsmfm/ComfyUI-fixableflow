# RGB Line Art Divider - 使用ガイド

## 概要
RGB Line Art Dividerは、線画と下塗り画像を入力として、下塗り画像のRGB値ごとに自動で領域分割を行い、レイヤー分けされたPSDファイルを生成するComfyUIノードです。

## 主な機能

### 🎨 **RGB Line Art Divider（基本版）**
線画と下塗り画像から簡単にレイヤー分けPSDを生成

**入力:**
- `line_art`: RGB値を持つ線画画像
- `base_color`: 下塗り画像（フラットカラー推奨）

**パラメータ:**
- `color_tolerance`: 同じ色と判定する許容値（0-50）
  - 低い値: より細かく色を分離
  - 高い値: 似た色をまとめる
- `line_blend_mode`: 線画レイヤーのブレンドモード
  - multiply: 乗算（デフォルト）
  - normal: 通常
  - darken: 比較（暗）
  - overlay: オーバーレイ
- `merge_small_regions`: 小さい領域をまとめるか
- `min_region_size`: 最小領域サイズ（ピクセル）

**出力:**
- `composite`: 合成済みプレビュー画像
- `base_color`: 下塗り画像
- `layer_count`: 生成されたレイヤー数
- `psd_path`: 保存されたPSDファイルのパス

### 🎯 **RGB Line Art Divider Advanced（詳細版）**
より細かい制御が可能な上級者向けバージョン

**追加パラメータ:**
- `line_opacity`: 線画の不透明度（0.0-1.0）
- `edge_smoothing`: エッジを滑らかにする
- `smoothing_kernel`: スムージングの強さ
- `separate_by_connectivity`: 同じ色でも離れた領域を別レイヤーに分ける
- `line_blend_mode`: screenモードも追加

**追加出力:**
- `region_preview`: 領域分割のプレビュー画像（各領域が異なる色で表示）

## 使用例

### 基本的な使い方
1. **線画を用意**: アンチエイリアスのないクリアな線画（黒線推奨）
2. **下塗り画像を用意**: フラットカラーで塗り分けた画像
3. **ノードに接続**: 両方の画像をノードに入力
4. **パラメータ調整**: 
   - 色の分離が細かすぎる場合は`color_tolerance`を上げる
   - 小さい領域が多い場合は`merge_small_regions`をONに
5. **実行**: PSDファイルが自動生成される

### 推奨設定

**アニメ塗り向け:**
```
color_tolerance: 5-10
line_blend_mode: multiply
merge_small_regions: True
min_region_size: 100
```

**イラスト向け:**
```
color_tolerance: 10-20
line_blend_mode: multiply
merge_small_regions: True
min_region_size: 200
```

**ピクセルアート向け:**
```
color_tolerance: 0-5
line_blend_mode: normal
merge_small_regions: False
separate_by_connectivity: True (Advancedのみ)
```

## PSDファイルの構造

生成されるPSDファイルは以下の構造になります：

```
├── Line Art（最上位レイヤー）
├── Color_R255_G100_B100（色領域1）
├── Color_R100_G255_B100（色領域2）
├── Color_R100_G100_B255（色領域3）
├── ...（その他の色領域）
└── Background（背景レイヤー）
```

各レイヤー名にはRGB値が含まれるため、どの色に対応するレイヤーかが一目でわかります。

## トラブルシューティング

### レイヤー数が多すぎる場合
- `color_tolerance`を上げて似た色をまとめる
- `merge_small_regions`をONにして小さい領域を統合
- `min_region_size`を大きくする

### 色が正しく分離されない場合
- 下塗り画像がアンチエイリアスされていないか確認
- グラデーションではなくフラットカラーを使用
- `color_tolerance`を下げてより厳密に色を判定

### 線画が見えない場合
- `line_blend_mode`を変更してみる
- Advancedノードで`line_opacity`を調整
- 線画の色が薄すぎないか確認

## 既存ノードとの違い

| 機能 | RGB Line Art Divider | 既存のLayerDivider |
|------|---------------------|-------------------|
| 入力画像数 | 2枚（線画+下塗り） | 1枚 |
| 分割方法 | RGB値ベース | クラスタリング |
| 処理速度 | 高速 | 中速 |
| 用途 | イラスト制作 | 写真処理 |
| PSD構造 | シンプル | 複雑 |

## 注意事項

- 入力画像は同じサイズである必要があります
- 下塗り画像はフラットカラー（単色塗り）が推奨されます
- グラデーションや複雑なテクスチャは適切に分離されない可能性があります
- 出力されるPSDファイルは`output`ディレクトリに保存されます
