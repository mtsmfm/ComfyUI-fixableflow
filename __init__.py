# RGB Line Art Dividerを使用
from .rgb_line_art_divider import RGB_NODE_CLASS_MAPPINGS, RGB_NODE_DISPLAY_NAME_MAPPINGS
from .rgb_line_art_divider_optimized import RGBLineArtDividerOptimized
from .layer_divider_simplified import NODE_CLASS_MAPPINGS as LEGACY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LEGACY_DISPLAY_MAPPINGS

# ノードマッピングを統合
NODE_CLASS_MAPPINGS = {
    **RGB_NODE_CLASS_MAPPINGS,  # 新しいRGBノードを優先
    "RGBLineArtDividerOptimized": RGBLineArtDividerOptimized,  # 最適化版を追加
    **LEGACY_MAPPINGS  # 既存のノードも保持
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RGB_NODE_DISPLAY_NAME_MAPPINGS,  # 新しいRGBノードを優先
    "RGBLineArtDividerOptimized": "RGB Line Art Divider (Optimized)",  # 最適化版を追加
    **LEGACY_DISPLAY_MAPPINGS  # 既存のノードも保持
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
