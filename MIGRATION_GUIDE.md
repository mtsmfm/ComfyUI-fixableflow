# ComfyUI-LayerDivider 簡略化版への移行ガイド

## 🎉 変更内容

このバージョンでは、コードベースを簡略化し、使いやすさを向上させました。

### 主な変更点

1. **セグメントマスク機能の削除**
   - `LayerDividerLoadMaskGenerator`ノードを削除
   - `LayerDividerSegmentMask`ノードを削除
   - SAM (Segment Anything Model) 関連の処理を削除

2. **新しいノード構成**
   - `LayerDivider` - オールインワンノード（1ステップで完結）
   - `LayerDivider - Advanced` - 詳細設定用
   - `LayerDivider - Divide` - レイヤー分割実行用

3. **依存関係の削減**
   - `segment_anything`ライブラリが不要に
   - `onnx`ライブラリが不要に
   - インストールが簡単になりました

## 📦 アップグレード方法

### 1. 依存関係の更新

```bash
# 不要なパッケージを削除
pip uninstall segment_anything onnx

# 必要なパッケージを確認
pip install -r requirements.txt
```

### 2. ComfyUIの再起動

ComfyUIを再起動して、新しいノード構成を読み込みます。

## 🔄 ワークフローの移行

### 旧ワークフロー（セグメントマスク使用）
```
Image → Load SAM → Segment Mask → Divide Layer
```

### 新ワークフロー（簡略版）
```
Image → LayerDivider (All-in-One)
```

または

```
Image → LayerDivider Advanced → LayerDivider Divide
```

## ✨ 新機能の使い方

### オールインワンノード

最もシンプルな使用方法：

1. `LayerDivider`ノードを追加
2. 画像を入力
3. パラメータを調整
4. 実行

一つのノードで全ての処理が完結します。

### アドバンスドモード

より詳細な制御が必要な場合：

1. `LayerDivider - Advanced`で解析
2. `LayerDivider - Divide`でレイヤー分割
3. 中間データの確認や調整が可能

## 📝 パラメータ説明

- **loops**: クラスタリングの反復回数（1-20）
- **init_cluster**: 初期クラスタ数（1-50）
- **ciede_threshold**: 色差の閾値（1-50）
- **blur_size**: ぼかしサイズ（1-20）
- **layer_mode**: 
  - `normal`: 通常レイヤー
  - `composite`: ブレンドモード付きレイヤー

## 🔧 トラブルシューティング

### エラーが発生する場合

1. ComfyUIを完全に終了
2. 以下のコマンドを実行：
   ```bash
   pip install --upgrade -r requirements.txt
   ```
3. ComfyUIを再起動

### 古いワークフローを使用する場合

オリジナル版に戻したい場合は、`__init__.py`を編集：

```python
# from .layer_divider_simplified import NODE_CLASS_MAPPINGS
from .layer_divider_node_original import NODE_CLASS_MAPPINGS
```

## 📚 補助ノード

以下の補助ノードは引き続き利用可能です：

- **Extract Line Art** - 線画抽出
- **Fill Area** - 領域塗りつぶし
- **Split Area** - 領域分割
- **Fill Space** - 隙間埋め

## 🆘 サポート

問題が発生した場合は、GitHubのIssuesにお問い合わせください。

---

**バージョン**: 2.0.0 (簡略化版)  
**更新日**: 2024
