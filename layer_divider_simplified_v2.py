"""
LayerDivider Simplified v2 - レイヤー分割専用版
カラーベース解析を削除し、前処理された画像をレイヤー分割するのみに特化
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .extract_lineart_node import ExtractLineArtNode, ExtractLineArtAdvancedNode
from .fill_area_node import FillAreaNode, FillAreaAdvancedNode
from .split_area_node import SplitAreaNode, SplitAreaAdvancedNode
from .fill_space_node import FillSpaceNode, FillSpaceAdvancedNode

import cv2

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


def split_into_layers(image, num_colors=5, blend_mode="normal"):
    """
    シンプルなレイヤー分割処理
    前処理された画像を受け取り、明度に基づいてレイヤーに分割
    """
    # NumPy配列に変換
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().detach().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.array(image)
    
    # 形状を調整
    if len(img_np.shape) == 4:
        img_np = img_np[0]
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
    
    # グレースケールに変換して明度を取得
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 明度に基づいてレイヤーを分割
    height, width = gray.shape
    base_layer = np.zeros((height, width, 4), dtype=np.uint8)
    bright_layer = np.zeros((height, width, 4), dtype=np.uint8)
    shadow_layer = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 閾値設定
    bright_threshold = 180
    shadow_threshold = 80
    
    for y in range(height):
        for x in range(width):
            pixel = img_np[y, x]
            brightness = gray[y, x]
            
            if brightness > bright_threshold:
                # 明るい部分
                bright_layer[y, x] = [pixel[0], pixel[1], pixel[2], 255]
                base_layer[y, x] = [128, 128, 128, 255]  # 中間色
            elif brightness < shadow_threshold:
                # 暗い部分
                shadow_layer[y, x] = [pixel[0], pixel[1], pixel[2], 255]
                base_layer[y, x] = [128, 128, 128, 255]  # 中間色
            else:
                # 中間部分（ベース）
                base_layer[y, x] = [pixel[0], pixel[1], pixel[2], 255]
    
    return [base_layer], [bright_layer], [shadow_layer]


def save_as_layers(base_layers, bright_layers, shadow_layers, filename_prefix="output"):
    """
    レイヤーを個別のPNG画像として保存
    PSD生成の代わりにシンプルな画像出力を提供
    """
    import uuid
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    base_filename = f"{filename_prefix}_{timestamp}_{unique_id}"
    
    saved_files = []
    
    # 各レイヤーを保存
    for i, layer in enumerate(base_layers):
        filepath = os.path.join(output_dir, f"{base_filename}_base_{i}.png")
        Image.fromarray(layer).save(filepath)
        saved_files.append(filepath)
    
    for i, layer in enumerate(bright_layers):
        filepath = os.path.join(output_dir, f"{base_filename}_bright_{i}.png")
        Image.fromarray(layer).save(filepath)
        saved_files.append(filepath)
    
    for i, layer in enumerate(shadow_layers):
        filepath = os.path.join(output_dir, f"{base_filename}_shadow_{i}.png")
        Image.fromarray(layer).save(filepath)
        saved_files.append(filepath)
    
    return base_filename


class LayerDividerSimple:
    """
    シンプルなレイヤー分割ノード
    前処理された画像を受け取り、レイヤーに分割
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_mode": (["luminance", "color_range", "simple"],),
                "num_layers": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "display": "slider"
                }),
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
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("original", "base", "bright", "shadow", "output_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, image, split_mode, num_layers, threshold_bright, threshold_shadow):
        # 入力画像を処理
        img_batch_np = image.cpu().detach().numpy()
        if img_batch_np.max() <= 1.0:
            img_batch_np = (img_batch_np * 255).astype(np.uint8)
        
        # 最初の画像を取得
        if len(img_batch_np.shape) == 4:
            input_image = img_batch_np[0]
        else:
            input_image = img_batch_np
        
        # RGBAに変換
        if input_image.shape[2] == 3:
            rgba_image = np.concatenate([
                input_image,
                np.ones((input_image.shape[0], input_image.shape[1], 1), dtype=np.uint8) * 255
            ], axis=2)
        else:
            rgba_image = input_image
        
        # 明度ベースの分割
        if split_mode == "luminance":
            gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            
            height, width = gray.shape
            base_layer = np.zeros((height, width, 4), dtype=np.uint8)
            bright_layer = np.zeros((height, width, 4), dtype=np.uint8)
            shadow_layer = np.zeros((height, width, 4), dtype=np.uint8)
            
            # マスクを作成
            bright_mask = gray > threshold_bright
            shadow_mask = gray < threshold_shadow
            base_mask = ~(bright_mask | shadow_mask)
            
            # 各レイヤーに割り当て
            base_layer[base_mask] = rgba_image[base_mask]
            bright_layer[bright_mask] = rgba_image[bright_mask]
            shadow_layer[shadow_mask] = rgba_image[shadow_mask]
            
            base_layers = [base_layer]
            bright_layers = [bright_layer]
            shadow_layers = [shadow_layer]
            
        elif split_mode == "simple":
            # シンプルな3分割
            base_layers, bright_layers, shadow_layers = split_into_layers(
                input_image, num_colors=num_layers, blend_mode="normal"
            )
        else:
            # カラーレンジモード（将来の拡張用）
            base_layers = [rgba_image.copy()]
            bright_layers = [np.zeros_like(rgba_image)]
            shadow_layers = [np.zeros_like(rgba_image)]
        
        # ファイル保存
        output_path = save_as_layers(base_layers, bright_layers, shadow_layers)
        
        # ComfyUI形式に変換して返す
        return (
            to_comfy_img(rgba_image),
            to_comfy_imgs(base_layers),
            to_comfy_imgs(bright_layers),
            to_comfy_imgs(shadow_layers),
            output_path
        )


class ImageToLayers:
    """
    画像を複数のレイヤーに分割する汎用ノード
    カラーベース解析は事前に他のノードで実行済みと想定
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "detail_image": ("IMAGE",),
            },
            "optional": {
                "shadow_image": ("IMAGE",),
                "highlight_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("combined", "layer1", "layer2", "layer3", "info")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"
    
    def execute(self, image, base_image, detail_image, shadow_image=None, highlight_image=None):
        """
        複数の処理済み画像を受け取り、レイヤーとして統合
        """
        # ベース画像の処理
        base_np = base_image.cpu().detach().numpy()
        if base_np.max() <= 1.0:
            base_np = (base_np * 255).astype(np.uint8)
        
        # ディテール画像の処理
        detail_np = detail_image.cpu().detach().numpy()
        if detail_np.max() <= 1.0:
            detail_np = (detail_np * 255).astype(np.uint8)
        
        # オプショナル画像の処理
        if shadow_image is not None:
            shadow_np = shadow_image.cpu().detach().numpy()
            if shadow_np.max() <= 1.0:
                shadow_np = (shadow_np * 255).astype(np.uint8)
        else:
            shadow_np = np.zeros_like(base_np)
        
        if highlight_image is not None:
            highlight_np = highlight_image.cpu().detach().numpy()
            if highlight_np.max() <= 1.0:
                highlight_np = (highlight_np * 255).astype(np.uint8)
        else:
            highlight_np = np.zeros_like(base_np)
        
        # 合成画像の作成
        combined = np.maximum(base_np, detail_np)
        combined = np.maximum(combined, shadow_np)
        combined = np.maximum(combined, highlight_np)
        
        # 情報テキスト
        info = f"Layers processed: base, detail"
        if shadow_image is not None:
            info += ", shadow"
        if highlight_image is not None:
            info += ", highlight"
        
        # ComfyUI形式に変換
        return (
            torch.from_numpy(combined.astype(np.float32) / 255.),
            base_image,
            detail_image,
            torch.from_numpy(shadow_np.astype(np.float32) / 255.),
            info
        )


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    # メインノード（シンプル版）
    "LayerDivider": LayerDividerSimple,
    
    # 汎用レイヤー統合ノード
    "Image To Layers": ImageToLayers,
    
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
    "LayerDivider": "Layer Divider (Simple)",
    "Image To Layers": "Image To Layers",
    "LayerDivider - Extract Line Art": "Extract Line Art",
    "LayerDivider - Extract Line Art Advanced": "Extract Line Art (Advanced)",
    "LayerDivider - Fill Area": "Fill Area",
    "LayerDivider - Fill Area Advanced": "Fill Area (Advanced)",
    "LayerDivider - Split Area": "Split Area",
    "LayerDivider - Split Area Advanced": "Split Area (Advanced)",
    "LayerDivider - Fill Space": "Fill Space",
    "LayerDivider - Fill Space Advanced": "Fill Space (Advanced)"
}
