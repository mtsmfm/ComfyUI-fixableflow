"""
LayerDivider Simplified v3 - PSD生成機能付きレイヤー分割版
カラーベース解析を削除し、前処理された画像をPSDレイヤーとして出力
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_utils import save_psd, divide_folder
from .ldivider.ld_convertor import pil2cv
from pytoshop.enums import BlendMode
from .extract_lineart_node import ExtractLineArtNode, ExtractLineArtAdvancedNode
from .fill_area_node import FillAreaNode, FillAreaAdvancedNode
from .split_area_node import SplitAreaNode, SplitAreaAdvancedNode
from .fill_space_node import FillSpaceNode, FillSpaceAdvancedNode

import cv2
import uuid
from datetime import datetime

# パス設定
comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'
output_dir = f"{layer_divider_path}/output"
input_dir = f"{layer_divider_path}/input"

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


def split_by_luminance(image, threshold_bright=180, threshold_shadow=80):
    """
    明度に基づいて画像をレイヤーに分割
    """
    # グレースケールに変換して明度を取得
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    height, width = gray.shape
    channels = 4 if image.shape[2] == 4 else 3
    
    # RGBA形式でレイヤーを作成
    base_layer = np.zeros((height, width, 4), dtype=np.uint8)
    bright_layer = np.zeros((height, width, 4), dtype=np.uint8)
    shadow_layer = np.zeros((height, width, 4), dtype=np.uint8)
    
    # マスクを作成
    bright_mask = gray > threshold_bright
    shadow_mask = gray < threshold_shadow
    base_mask = ~(bright_mask | shadow_mask)
    
    # 元画像をRGBAに変換
    if channels == 3:
        rgba_image = np.concatenate([
            image,
            np.ones((height, width, 1), dtype=np.uint8) * 255
        ], axis=2)
    else:
        rgba_image = image
    
    # 各レイヤーに割り当て
    base_layer[base_mask] = rgba_image[base_mask]
    bright_layer[bright_mask] = rgba_image[bright_mask]
    shadow_layer[shadow_mask] = rgba_image[shadow_mask]
    
    return [base_layer], [bright_layer], [shadow_layer]


def split_by_color_range(image, num_colors=5):
    """
    色範囲に基づいて画像をレイヤーに分割（シンプルな量子化）
    """
    from sklearn.cluster import KMeans
    
    # 画像を2D配列に変換
    h, w, c = image.shape
    image_2d = image.reshape(-1, 3 if c >= 3 else 1)
    
    # K-meansでクラスタリング
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(image_2d).reshape(h, w)
    
    # 明度でソート
    centers = kmeans.cluster_centers_
    brightness = np.mean(centers, axis=1)
    sorted_indices = np.argsort(brightness)
    
    # レイヤーを作成
    base_layers = []
    bright_layers = []
    shadow_layers = []
    
    # 各クラスタをレイヤーに割り当て
    for i, idx in enumerate(sorted_indices):
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        mask = labels == idx
        
        if c >= 3:
            layer[mask, :3] = image[mask, :3]
        else:
            layer[mask, :3] = np.repeat(image[mask, :1], 3, axis=1)
        layer[mask, 3] = 255
        
        # 明度に基づいて分類
        if i < len(sorted_indices) // 3:
            shadow_layers.append(layer)
        elif i < 2 * len(sorted_indices) // 3:
            base_layers.append(layer)
        else:
            bright_layers.append(layer)
    
    # 各カテゴリが空でないことを保証
    if not base_layers:
        base_layers = [np.zeros((h, w, 4), dtype=np.uint8)]
    if not bright_layers:
        bright_layers = [np.zeros((h, w, 4), dtype=np.uint8)]
    if not shadow_layers:
        shadow_layers = [np.zeros((h, w, 4), dtype=np.uint8)]
    
    return base_layers, bright_layers, shadow_layers


class LayerDividerWithPSD:
    """
    前処理された画像をレイヤーに分割してPSDファイルを生成
    カラーベース解析は他のノードで実行済みと想定
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_mode": (["luminance", "color_range", "direct"],),
                "layer_mode": (["normal", "composite"],),
                "threshold_bright": ("INT", {
                    "default": 180,
                    "min": 100,
                    "max": 255,
                    "step": 5,
                    "display": "slider"
                }),
                "threshold_shadow": ("INT", {
                    "default": 80,
                    "min": 0,
                    "max": 150,
                    "step": 5,
                    "display": "slider"
                }),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "base_layer": ("IMAGE",),
                "bright_layer": ("IMAGE",),
                "shadow_layer": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("original", "base", "bright", "shadow", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, image, split_mode, layer_mode, threshold_bright, threshold_shadow, 
                num_colors, base_layer=None, bright_layer=None, shadow_layer=None):
        
        # 入力画像を処理
        img_batch_np = image.cpu().detach().numpy()
        if img_batch_np.max() <= 1.0:
            img_batch_np = (img_batch_np * 255).astype(np.uint8)
        
        # 最初の画像を取得
        if len(img_batch_np.shape) == 4:
            input_image = img_batch_np[0]
        else:
            input_image = img_batch_np
        
        # OpenCV形式に変換（PSD生成用）
        if input_image.shape[2] == 3:
            input_image_cv = cv2.cvtColor(input_image, cv2.COLOR_RGB2RGBA)
        else:
            input_image_cv = input_image
        
        # レイヤー分割処理
        if split_mode == "direct" and base_layer is not None:
            # 直接入力されたレイヤーを使用
            def process_layer_input(layer_input):
                if layer_input is None:
                    return [np.zeros_like(input_image_cv)]
                layer_np = layer_input.cpu().detach().numpy()
                if layer_np.max() <= 1.0:
                    layer_np = (layer_np * 255).astype(np.uint8)
                if len(layer_np.shape) == 4:
                    layer_np = layer_np[0]
                if layer_np.shape[2] == 3:
                    layer_np = cv2.cvtColor(layer_np, cv2.COLOR_RGB2RGBA)
                return [layer_np]
            
            base_layer_list = process_layer_input(base_layer)
            bright_layer_list = process_layer_input(bright_layer)
            shadow_layer_list = process_layer_input(shadow_layer)
            
        elif split_mode == "luminance":
            # 明度ベースの分割
            base_layer_list, bright_layer_list, shadow_layer_list = split_by_luminance(
                input_image, threshold_bright, threshold_shadow
            )
            
        elif split_mode == "color_range":
            # 色範囲ベースの分割
            base_layer_list, bright_layer_list, shadow_layer_list = split_by_color_range(
                input_image, num_colors
            )
        else:
            # デフォルト：そのまま使用
            base_layer_list = [input_image_cv]
            bright_layer_list = [np.zeros_like(input_image_cv)]
            shadow_layer_list = [np.zeros_like(input_image_cv)]
        
        # PSDファイル生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        if layer_mode == "composite":
            # コンポジットモード（ブレンドモード付き）
            # リストを調整（PSD生成関数の期待する形式に）
            addition_layer_list = [np.zeros_like(input_image_cv)]
            subtract_layer_list = [np.zeros_like(input_image_cv)]
            
            filename = save_psd(
                input_image_cv,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
                "simplified"
            )
        else:
            # ノーマルモード
            filename = save_psd(
                input_image_cv,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
                "simplified"
            )
        
        print(f"PSD file saved: {filename}")
        
        # フォルダに分割
        divide_folder(filename, input_dir, layer_mode)
        
        # ComfyUI形式に変換して返す
        return (
            to_comfy_img(input_image_cv),
            to_comfy_imgs(base_layer_list),
            to_comfy_imgs(bright_layer_list),
            to_comfy_imgs(shadow_layer_list),
            filename
        )


class ProcessedImageToPSD:
    """
    前処理済みの複数画像を受け取ってPSDファイルを生成
    各種処理ノードの出力を直接PSD化
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "layer_mode": (["normal", "composite"],),
            },
            "optional": {
                "base_layer": ("IMAGE",),
                "bright_layer": ("IMAGE",),
                "shadow_layer": ("IMAGE",),
                "detail_layer": ("IMAGE",),
                "line_layer": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("combined", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"
    
    def execute(self, original_image, layer_mode, base_layer=None, bright_layer=None, 
                shadow_layer=None, detail_layer=None, line_layer=None):
        """
        複数の処理済み画像からPSDファイルを生成
        """
        # 元画像の処理
        orig_np = original_image.cpu().detach().numpy()
        if orig_np.max() <= 1.0:
            orig_np = (orig_np * 255).astype(np.uint8)
        if len(orig_np.shape) == 4:
            orig_np = orig_np[0]
        if orig_np.shape[2] == 3:
            orig_cv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2RGBA)
        else:
            orig_cv = orig_np
        
        # レイヤーリストの準備
        all_layers = []
        layer_names = []
        blend_modes = []
        
        # 各レイヤーを処理して追加
        def add_layer(layer_input, name, blend_mode=BlendMode.normal):
            if layer_input is not None:
                layer_np = layer_input.cpu().detach().numpy()
                if layer_np.max() <= 1.0:
                    layer_np = (layer_np * 255).astype(np.uint8)
                if len(layer_np.shape) == 4:
                    layer_np = layer_np[0]
                if layer_np.shape[2] == 3:
                    layer_np = cv2.cvtColor(layer_np, cv2.COLOR_RGB2RGBA)
                all_layers.append([layer_np])
                layer_names.append(name)
                blend_modes.append(blend_mode)
        
        # レイヤーを追加
        if layer_mode == "composite":
            # コンポジットモード
            add_layer(base_layer, "base", BlendMode.normal)
            add_layer(bright_layer, "screen", BlendMode.screen)
            add_layer(shadow_layer, "multiply", BlendMode.multiply)
            add_layer(detail_layer, "overlay", BlendMode.overlay)
            add_layer(line_layer, "linear_burn", BlendMode.linear_burn)
        else:
            # ノーマルモード
            add_layer(base_layer, "base", BlendMode.normal)
            add_layer(bright_layer, "bright", BlendMode.normal)
            add_layer(shadow_layer, "shadow", BlendMode.normal)
            add_layer(detail_layer, "detail", BlendMode.normal)
            add_layer(line_layer, "line", BlendMode.normal)
        
        # 少なくとも1つのレイヤーが必要
        if not all_layers:
            all_layers.append([orig_cv])
            layer_names.append("original")
            blend_modes.append(BlendMode.normal)
        
        # PSD生成
        filename = save_psd(
            orig_cv,
            all_layers,
            layer_names,
            blend_modes,
            output_dir,
            layer_mode,
            "processed"
        )
        
        print(f"PSD file saved: {filename}")
        
        # 合成画像の作成（プレビュー用）
        combined = orig_cv.copy()
        for layers in all_layers:
            for layer in layers:
                # シンプルな合成（より高度な合成はPSDエディタで）
                mask = layer[:, :, 3] > 0
                combined[mask] = layer[mask]
        
        return (
            to_comfy_img(combined),
            filename
        )


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    # メインノード（PSD生成機能付き）
    "LayerDivider": LayerDividerWithPSD,
    
    # 処理済み画像をPSD化するノード
    "Processed Image to PSD": ProcessedImageToPSD,
    
    # 線画処理ノード
    "LayerDivider - Extract Line Art": ExtractLineArtNode,
    "LayerDivider - Extract Line Art Advanced": ExtractLineArtAdvancedNode,
    
    # 領域処理ノード
    "LayerDivider - Fill Area": FillAreaNode,
    "LayerDivider - Fill Area Advanced": FillAreaAdvancedNode,
    "LayerDivider - Split Area": SplitAreaNode,
    "LayerDivider - Split Area Advanced": SplitAreaAdvancedNode,
    "LayerDivider - Fill Space": FillSpaceNode,
    "LayerDivider - Fill Space Advanced": FillSpaceAdvancedNode
}

# 表示名マッピング
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerDivider": "Layer Divider with PSD",
    "Processed Image to PSD": "Processed Image to PSD",
    "LayerDivider - Extract Line Art": "Extract Line Art",
    "LayerDivider - Extract Line Art Advanced": "Extract Line Art (Advanced)",
    "LayerDivider - Fill Area": "Fill Area",
    "LayerDivider - Fill Area Advanced": "Fill Area (Advanced)",
    "LayerDivider - Split Area": "Split Area",
    "LayerDivider - Split Area Advanced": "Split Area (Advanced)",
    "LayerDivider - Fill Space": "Fill Space",
    "LayerDivider - Fill Space Advanced": "Fill Space (Advanced)"
}
