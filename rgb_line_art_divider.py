"""
RGB Line Art Layer Divider
線画と下塗り画像を入力として、RGB値ごとに領域分割してPSDを生成するノード
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_utils import save_psd
from .ldivider.ld_convertor import pil2cv
from pytoshop.enums import BlendMode
import cv2
from collections import defaultdict
import pytoshop
from pytoshop.core import PsdFile
from pytoshop.layers import Layer, PixelLayer

# パス設定
comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'
output_dir = f"{layer_divider_path}/output"

if not os.path.exists(f'{output_dir}'):
    os.makedirs(f'{output_dir}')


def HWC3(x):
    """画像形式の変換ヘルパー関数"""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def to_comfy_img(np_img):
    """NumPy配列をComfyUI形式の画像に変換"""
    out_imgs = []
    out_imgs.append(HWC3(np_img))
    out_imgs = np.stack(out_imgs)
    out_imgs = torch.from_numpy(out_imgs.astype(np.float32) / 255.)
    return out_imgs


def to_comfy_imgs(np_imgs):
    """複数のNumPy配列をComfyUI形式の画像バッチに変換"""
    out_imgs = []
    for np_img in np_imgs:
        out_imgs.append(HWC3(np_img))
    out_imgs = np.stack(out_imgs)
    out_imgs = torch.from_numpy(out_imgs.astype(np.float32) / 255.)
    return out_imgs


def extract_color_regions(base_image_cv, tolerance=10):
    """
    下塗り画像からRGB値ごとに領域を抽出
    
    Args:
        base_image_cv: 下塗り画像（BGRA形式）
        tolerance: 同じ色と判定する許容値
    
    Returns:
        color_regions: {(R,G,B): mask} の辞書
    """
    # BGRAからBGRに変換（アルファチャンネルを考慮）
    if base_image_cv.shape[2] == 4:
        bgr_image = base_image_cv[:, :, :3]
        alpha = base_image_cv[:, :, 3]
    else:
        bgr_image = base_image_cv
        alpha = np.ones((base_image_cv.shape[0], base_image_cv.shape[1]), dtype=np.uint8) * 255
    
    # RGBに変換（OpenCVはBGRなので）
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # ユニークな色を取得
    height, width = rgb_image.shape[:2]
    pixels = rgb_image.reshape(-1, 3)
    alpha_flat = alpha.reshape(-1)
    
    # アルファ値が0でないピクセルのみを対象とする
    valid_pixels = pixels[alpha_flat > 0]
    
    # 色をグループ化（tolerance考慮）
    color_regions = defaultdict(list)
    processed = set()
    
    for i, pixel in enumerate(valid_pixels):
        if i in processed:
            continue
            
        # tolerance範囲内の色を同じグループとして扱う
        color_key = tuple(pixel)
        similar_indices = []
        
        for j, other_pixel in enumerate(valid_pixels):
            if j not in processed:
                diff = np.abs(pixel.astype(int) - other_pixel.astype(int))
                if np.all(diff <= tolerance):
                    similar_indices.append(j)
                    processed.add(j)
        
        if similar_indices:
            # 平均色を計算してキーとする
            similar_colors = valid_pixels[similar_indices]
            avg_color = np.mean(similar_colors, axis=0).astype(int)
            color_key = tuple(avg_color)
            
            # マスクを作成
            mask = np.zeros((height, width), dtype=np.uint8)
            for idx in similar_indices:
                # 元のインデックスを復元
                original_idx = np.where(alpha_flat > 0)[0][idx]
                y = original_idx // width
                x = original_idx % width
                mask[y, x] = 255
            
            color_regions[color_key] = mask
    
    # 実際の領域マスクを作成
    final_regions = {}
    for color, indices in color_regions.items():
        if isinstance(indices, np.ndarray):
            final_regions[color] = indices
        else:
            # インデックスからマスクを作成
            mask = np.zeros((height, width), dtype=np.uint8)
            for idx in indices:
                mask.flat[idx] = 255
            final_regions[color] = mask
    
    return final_regions


def create_region_layers(base_image_cv, color_regions):
    """
    色領域ごとにレイヤーを作成
    
    Args:
        base_image_cv: 元画像（BGRA形式）
        color_regions: {(R,G,B): mask} の辞書
    
    Returns:
        layers: レイヤーリスト
        names: レイヤー名リスト
    """
    layers = []
    names = []
    
    for color, mask in color_regions.items():
        # マスクを適用してレイヤーを作成
        layer = np.zeros_like(base_image_cv)
        
        # マスクがある部分だけ色を適用
        mask_3d = np.stack([mask] * 4, axis=2)
        layer[mask_3d > 0] = base_image_cv[mask_3d > 0]
        
        layers.append(layer)
        # レイヤー名をRGB値で作成
        names.append(f"Color_R{color[0]}_G{color[1]}_B{color[2]}")
    
    return layers, names


def save_psd_with_lineart(base_image_cv, line_art_cv, color_layers, layer_names, output_dir, filename_prefix="rgb_divided"):
    """
    線画を最上位レイヤーとしてPSDファイルを保存
    
    Args:
        base_image_cv: ベース画像
        line_art_cv: 線画（BGRA形式）
        color_layers: 色領域レイヤーのリスト
        layer_names: レイヤー名のリスト
        output_dir: 出力ディレクトリ
        filename_prefix: ファイル名プレフィックス
    
    Returns:
        filename: 保存したファイル名
    """
    import time
    from datetime import datetime
    
    # ファイル名生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.psd")
    
    # PSDファイルの作成
    height, width = base_image_cv.shape[:2]
    psd = PsdFile(width=width, height=height, depth=8, color_mode=3)  # RGB mode
    
    # 背景レイヤーを追加
    bg_layer = PixelLayer(name="Background", width=width, height=height)
    bg_data = base_image_cv[:, :, [2, 1, 0, 3]]  # BGRA to RGBA
    bg_layer.set_data(bg_data)
    psd.layers.append(bg_layer)
    
    # 色領域レイヤーを追加
    for layer_data, name in zip(color_layers, layer_names):
        layer = PixelLayer(name=name, width=width, height=height)
        # BGRAからRGBAに変換
        rgba_data = layer_data[:, :, [2, 1, 0, 3]]
        layer.set_data(rgba_data)
        layer.blend_mode = BlendMode.normal
        psd.layers.append(layer)
    
    # 線画レイヤーを最上位に追加
    line_layer = PixelLayer(name="Line Art", width=width, height=height)
    line_rgba = line_art_cv[:, :, [2, 1, 0, 3]]  # BGRA to RGBA
    line_layer.set_data(line_rgba)
    line_layer.blend_mode = BlendMode.multiply  # 線画は乗算モードが一般的
    psd.layers.append(line_layer)
    
    # ファイルを保存
    with open(filename, 'wb') as f:
        psd.write(f)
    
    return filename


class RGBLineArtDivider:
    """
    RGB線画と下塗り画像から領域分割PSDを生成するノード
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "line_art": ("IMAGE",),
                "base_color": ("IMAGE",),
                "color_tolerance": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "line_blend_mode": (["multiply", "normal", "darken", "overlay"],),
                "merge_small_regions": ("BOOLEAN", {
                    "default": True
                }),
                "min_region_size": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("composite", "base_color", "layer_count", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, line_art, base_color, color_tolerance, line_blend_mode, 
                merge_small_regions, min_region_size):
        
        # 画像をNumPy配列に変換
        line_art_np = line_art.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
        base_color_np = base_color.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
        
        # PIL Imageに変換
        line_art_pil = Image.fromarray(line_art_np)
        base_color_pil = Image.fromarray(base_color_np)
        
        # OpenCV形式（BGRA）に変換
        line_art_cv = pil2cv(line_art_pil)
        base_color_cv = pil2cv(base_color_pil)
        
        # BGRAに変換（アルファチャンネルを追加）
        if line_art_cv.shape[2] == 3:
            line_art_cv = cv2.cvtColor(line_art_cv, cv2.COLOR_BGR2BGRA)
        if base_color_cv.shape[2] == 3:
            base_color_cv = cv2.cvtColor(base_color_cv, cv2.COLOR_BGR2BGRA)
        
        # 色領域を抽出
        color_regions = extract_color_regions(base_color_cv, tolerance=color_tolerance)
        
        # 小さい領域をマージ（オプション）
        if merge_small_regions:
            filtered_regions = {}
            small_region_mask = np.zeros(base_color_cv.shape[:2], dtype=np.uint8)
            
            for color, mask in color_regions.items():
                region_size = np.sum(mask > 0)
                if region_size >= min_region_size:
                    filtered_regions[color] = mask
                else:
                    small_region_mask = cv2.bitwise_or(small_region_mask, mask)
            
            # 小さい領域を「その他」としてまとめる
            if np.any(small_region_mask > 0):
                filtered_regions[(128, 128, 128)] = small_region_mask  # グレーとして追加
            
            color_regions = filtered_regions
        
        # レイヤーを作成
        color_layers, layer_names = create_region_layers(base_color_cv, color_regions)
        
        # BlendModeの設定
        blend_mode_map = {
            "multiply": BlendMode.multiply,
            "normal": BlendMode.normal,
            "darken": BlendMode.darken,
            "overlay": BlendMode.overlay
        }
        
        # PSDファイルを保存（既存のsave_psd関数を使用）
        all_layers = [color_layers, [line_art_cv]]
        all_names = layer_names + ["Line Art"]
        all_modes = [BlendMode.normal] * len(layer_names) + [blend_mode_map[line_blend_mode]]
        
        filename = save_psd(
            base_color_cv,
            all_layers,
            all_names,
            all_modes,
            output_dir,
            "normal",
            "rgb_divided"
        )
        
        print(f"PSD file saved: {filename}")
        print(f"Created {len(color_regions)} color region layers")
        
        # コンポジット画像を作成（プレビュー用）
        composite = base_color_cv.copy()
        if line_blend_mode == "multiply":
            # 乗算合成
            line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
            composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
            composite[:, :, :3] = (composite_rgb * line_rgb * 255).astype(np.uint8)
        elif line_blend_mode == "normal":
            # 線画のアルファを考慮して合成
            alpha = line_art_cv[:, :, 3:4].astype(np.float32) / 255.0
            composite[:, :, :3] = (
                line_art_cv[:, :, :3] * alpha + 
                composite[:, :, :3] * (1 - alpha)
            ).astype(np.uint8)
        
        # 出力
        return (
            to_comfy_img(composite),
            to_comfy_img(base_color_cv),
            len(color_regions),
            filename
        )


class RGBLineArtDividerAdvanced:
    """
    より詳細な設定が可能なRGB線画分割ノード
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "line_art": ("IMAGE",),
                "base_color": ("IMAGE",),
                "color_tolerance": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "line_blend_mode": (["multiply", "normal", "darken", "overlay", "screen"],),
                "line_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "merge_small_regions": ("BOOLEAN", {
                    "default": True
                }),
                "min_region_size": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "slider"
                }),
                "edge_smoothing": ("BOOLEAN", {
                    "default": False
                }),
                "smoothing_kernel": ("INT", {
                    "default": 3,
                    "min": 3,
                    "max": 15,
                    "step": 2,
                    "display": "slider"
                }),
                "separate_by_connectivity": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("composite", "base_color", "region_preview", "layer_count", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, line_art, base_color, color_tolerance, line_blend_mode, 
                line_opacity, merge_small_regions, min_region_size, 
                edge_smoothing, smoothing_kernel, separate_by_connectivity):
        
        # 画像をNumPy配列に変換
        line_art_np = line_art.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
        base_color_np = base_color.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
        
        # PIL Imageに変換
        line_art_pil = Image.fromarray(line_art_np)
        base_color_pil = Image.fromarray(base_color_np)
        
        # OpenCV形式（BGRA）に変換
        line_art_cv = pil2cv(line_art_pil)
        base_color_cv = pil2cv(base_color_pil)
        
        # BGRAに変換
        if line_art_cv.shape[2] == 3:
            line_art_cv = cv2.cvtColor(line_art_cv, cv2.COLOR_BGR2BGRA)
        if base_color_cv.shape[2] == 3:
            base_color_cv = cv2.cvtColor(base_color_cv, cv2.COLOR_BGR2BGRA)
        
        # 線画の不透明度を調整
        if line_opacity < 1.0:
            line_art_cv[:, :, 3] = (line_art_cv[:, :, 3] * line_opacity).astype(np.uint8)
        
        # 色領域を抽出
        color_regions = extract_color_regions(base_color_cv, tolerance=color_tolerance)
        
        # 連結成分で分離（オプション）
        if separate_by_connectivity:
            new_regions = {}
            region_counter = 0
            
            for color, mask in color_regions.items():
                # 連結成分を検出
                num_labels, labels = cv2.connectedComponents(mask)
                
                for label_id in range(1, num_labels):  # 0は背景
                    component_mask = (labels == label_id).astype(np.uint8) * 255
                    
                    # 小さい領域のフィルタリング
                    if merge_small_regions and np.sum(component_mask > 0) < min_region_size:
                        continue
                    
                    # 新しい領域として追加
                    new_color = (color[0], color[1], color[2], region_counter)  # IDを付加
                    new_regions[new_color] = component_mask
                    region_counter += 1
            
            color_regions = new_regions
        
        # エッジスムージング（オプション）
        if edge_smoothing:
            smoothed_regions = {}
            for color, mask in color_regions.items():
                # モルフォロジー処理でエッジを滑らかに
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                  (smoothing_kernel, smoothing_kernel))
                smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
                smoothed_regions[color] = smoothed_mask
            color_regions = smoothed_regions
        
        # 小さい領域をマージ（separate_by_connectivityと併用可能）
        if merge_small_regions and not separate_by_connectivity:
            filtered_regions = {}
            small_region_mask = np.zeros(base_color_cv.shape[:2], dtype=np.uint8)
            
            for color, mask in color_regions.items():
                region_size = np.sum(mask > 0)
                if region_size >= min_region_size:
                    filtered_regions[color] = mask
                else:
                    small_region_mask = cv2.bitwise_or(small_region_mask, mask)
            
            if np.any(small_region_mask > 0):
                filtered_regions[(128, 128, 128)] = small_region_mask
            
            color_regions = filtered_regions
        
        # 領域プレビュー画像を作成
        region_preview = np.zeros_like(base_color_cv)
        color_palette = []
        
        # カラーパレットを生成（各領域に異なる色を割り当て）
        np.random.seed(42)  # 再現性のため
        for i in range(len(color_regions)):
            color = [
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                255
            ]
            color_palette.append(color)
        
        # プレビュー画像に色を適用
        for (color, mask), palette_color in zip(color_regions.items(), color_palette):
            mask_3d = np.stack([mask > 0] * 4, axis=2)
            region_preview[mask_3d] = palette_color * (mask_3d[mask_3d].reshape(-1) // 4)
        
        # レイヤーを作成
        color_layers, layer_names = create_region_layers(base_color_cv, color_regions)
        
        # BlendModeの設定
        blend_mode_map = {
            "multiply": BlendMode.multiply,
            "normal": BlendMode.normal,
            "darken": BlendMode.darken,
            "overlay": BlendMode.overlay,
            "screen": BlendMode.screen
        }
        
        # PSDファイルを保存
        all_layers = [color_layers, [line_art_cv]]
        all_names = layer_names + ["Line Art"]
        all_modes = [BlendMode.normal] * len(layer_names) + [blend_mode_map[line_blend_mode]]
        
        filename = save_psd(
            base_color_cv,
            all_layers,
            all_names,
            all_modes,
            output_dir,
            "normal",
            "rgb_divided_advanced"
        )
        
        print(f"PSD file saved: {filename}")
        print(f"Created {len(color_regions)} color region layers")
        
        # コンポジット画像を作成
        composite = base_color_cv.copy()
        
        # 線画の合成
        if line_blend_mode == "multiply":
            line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
            composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
            composite[:, :, :3] = (composite_rgb * line_rgb * 255).astype(np.uint8)
        elif line_blend_mode == "screen":
            line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
            composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
            composite[:, :, :3] = ((1 - (1 - composite_rgb) * (1 - line_rgb)) * 255).astype(np.uint8)
        elif line_blend_mode == "overlay":
            line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
            composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
            overlay_result = np.where(
                composite_rgb < 0.5,
                2 * composite_rgb * line_rgb,
                1 - 2 * (1 - composite_rgb) * (1 - line_rgb)
            )
            composite[:, :, :3] = (overlay_result * 255).astype(np.uint8)
        elif line_blend_mode == "darken":
            composite[:, :, :3] = np.minimum(composite[:, :, :3], line_art_cv[:, :, :3])
        else:  # normal
            alpha = line_art_cv[:, :, 3:4].astype(np.float32) / 255.0
            composite[:, :, :3] = (
                line_art_cv[:, :, :3] * alpha + 
                composite[:, :, :3] * (1 - alpha)
            ).astype(np.uint8)
        
        # 出力
        return (
            to_comfy_img(composite),
            to_comfy_img(base_color_cv),
            to_comfy_img(region_preview),
            len(color_regions),
            filename
        )


# ノードマッピング用の辞書を作成
RGB_NODE_CLASS_MAPPINGS = {
    "RGBLineArtDivider": RGBLineArtDivider,
    "RGBLineArtDividerAdvanced": RGBLineArtDividerAdvanced,
}

RGB_NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBLineArtDivider": "RGB Line Art Divider",
    "RGBLineArtDividerAdvanced": "RGB Line Art Divider (Advanced)",
}
