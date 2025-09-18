"""
LayerDivider Node - Color Base Only Version
セグメントマスク機能を削除し、カラーベースのみに特化したバージョン
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_utils import save_psd, divide_folder
from .ldivider.ld_convertor import pil2cv, cv2pil, df2bgra
from .ldivider.ld_processor import get_base, get_normal_layer, get_composite_layer
from pytoshop.enums import BlendMode
from .extract_lineart_node import ExtractLineArtNode, ExtractLineArtAdvancedNode
from .fill_area_node import FillAreaNode, FillAreaAdvancedNode
from .split_area_node import SplitAreaNode, SplitAreaAdvancedNode
from .fill_space_node import FillSpaceNode, FillSpaceAdvancedNode

import cv2

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


class LayerDividerColorBase:
    """カラーベースでレイヤー分割用のデータを生成するノード"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "loops": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "init_cluster": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "ciede_threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "blur_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("LD_INPUT_IMAGE", "LD_DF")
    RETURN_NAMES = ("input_image", "df")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider/ColorBase"

    def execute(self, image, loops, init_cluster, ciede_threshold, blur_size):
        # 背景除去は無効化
        split_bg = False
        h_split = -1
        v_split = -1
        n_cluster = -1
        alpha = -1
        th_rate = 0

        # ComfyUI形式の画像をNumPy配列に変換
        img_batch_np = image.cpu().detach().numpy().__mul__(255.).astype(np.uint8)
        input_image = Image.fromarray(img_batch_np[0])
        
        # OpenCV形式に変換
        image = pil2cv(input_image)
        input_image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        # カラーベース処理を実行
        df = get_base(
            input_image_cv, loops, init_cluster, ciede_threshold, blur_size,
            h_split, v_split, n_cluster, alpha, th_rate, split_bg, False
        )

        return input_image_cv, df


class LayerDividerDivideLayer:
    """レイヤー分割を実行してPSDファイルを出力するノード"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("LD_INPUT_IMAGE",),
                "df": ("LD_DF",),
                "layer_mode": (["composite", "normal"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("base_image", "base", "bright", "shadow", "filepath")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider/ColorBase"

    def execute(self, input_image, df, layer_mode):
        # divide_modeは常に"color_base"として扱う
        divide_mode = "color_base"
        
        if layer_mode == "composite":
            # コンポジットモードのレイヤー生成
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = (
                get_composite_layer(input_image, df)
            )
            
            filename = save_psd(
                input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
                divide_mode
            )

        elif layer_mode == "normal":
            # ノーマルモードのレイヤー生成
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
            
            filename = save_psd(
                input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
                divide_mode
            )

        print(f"PSD file saved: {filename}")
        
        # フォルダに分割
        divide_folder(filename, input_dir, layer_mode)

        # ComfyUI形式に変換して返す
        return (
            to_comfy_img(input_image),
            to_comfy_imgs(base_layer_list),
            to_comfy_imgs(bright_layer_list),
            to_comfy_imgs(shadow_layer_list),
            filename
        )


# ノードクラスマッピング（カラーベースのみ）
NODE_CLASS_MAPPINGS = {
    "LayerDivider - Color Base Only": LayerDividerColorBase,
    "LayerDivider - Divide Layer (Color Base)": LayerDividerDivideLayer,
    "LayerDivider - Extract Line Art": ExtractLineArtNode,
    "LayerDivider - Extract Line Art Advanced": ExtractLineArtAdvancedNode,
    "LayerDivider - Fill Area": FillAreaNode,
    "LayerDivider - Fill Area Advanced": FillAreaAdvancedNode,
    "LayerDivider - Split Area": SplitAreaNode,
    "LayerDivider - Split Area Advanced": SplitAreaAdvancedNode,
    "LayerDivider - Fill Space": FillSpaceNode,
    "LayerDivider - Fill Space Advanced": FillSpaceAdvancedNode
}

# ノード表示名マッピング
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerDivider - Color Base Only": "Color Base (Simplified)",
    "LayerDivider - Divide Layer (Color Base)": "Divide Layer (Color Base)",
    "LayerDivider - Extract Line Art": "Extract Line Art",
    "LayerDivider - Extract Line Art Advanced": "Extract Line Art (Advanced)",
    "LayerDivider - Fill Area": "Fill Area",
    "LayerDivider - Fill Area Advanced": "Fill Area (Advanced)",
    "LayerDivider - Split Area": "Split Area",
    "LayerDivider - Split Area Advanced": "Split Area (Advanced)",
    "LayerDivider - Fill Space": "Fill Space",
    "LayerDivider - Fill Space Advanced": "Fill Space (Advanced)"
}
