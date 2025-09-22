# RGB Line Art Dividerを使用 - Updated to force reload
from .rgb_line_art_divider import RGB_NODE_CLASS_MAPPINGS, RGB_NODE_DISPLAY_NAME_MAPPINGS

# RGBLineArtDividerFastの最新版をインポート
from .rgb_line_art_divider_fast import RGBLineArtDividerFast

from .layer_divider_simplified import NODE_CLASS_MAPPINGS as LEGACY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LEGACY_DISPLAY_MAPPINGS

# 改良版FillSpaceNodeをインポート
from .fill_space_improved_node import NODE_CLASS_MAPPINGS as FILL_IMPROVED_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FILL_IMPROVED_DISPLAY_MAPPINGS

# ノードマッピングを統合
NODE_CLASS_MAPPINGS = {
    **RGB_NODE_CLASS_MAPPINGS,  # 新しいRGBノードを優先
    "RGBLineArtDividerFast": RGBLineArtDividerFast,  # 高速版を追加（更新版）
    **LEGACY_MAPPINGS,  # 既存のノードも保持
    **FILL_IMPROVED_MAPPINGS  # 改良版FillSpaceノードを追加
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RGB_NODE_DISPLAY_NAME_MAPPINGS,  # 新しいRGBノードを優先
    "RGBLineArtDividerFast": "RGB Line Art Divider (Fast)",  # 高速版を追加
    **LEGACY_DISPLAY_MAPPINGS,  # 既存のノードも保持
    **FILL_IMPROVED_DISPLAY_MAPPINGS  # 改良版FillSpaceノードを追加
}

# Web拡張機能の自動読み込みのためのパス設定
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
