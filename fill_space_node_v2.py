"""
Fill Space Node V2 for ComfyUI
線画の下のピクセルを、クラスタ化された色から最も近い色で塗りつぶすノード
CIEDE2000色差計算を使用
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from skimage import color as skcolor
import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-fixableflow'
output_dir = f"{layer_divider_path}/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def tensor_to_pil(tensor):
    """ComfyUIのテンソル形式をPIL Imageに変換"""
    image_np = tensor.cpu().detach().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # バッチの最初の画像を取得
    image_np = (image_np * 255).astype(np.uint8)
    
    if image_np.shape[2] == 3:
        mode = 'RGB'
    elif image_np.shape[2] == 4:
        mode = 'RGBA'
    else:
        mode = 'L'
    
    return Image.fromarray(image_np, mode=mode)


def pil_to_tensor(image):
    """PIL ImageをComfyUIのテンソル形式に変換"""
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)


def rgb_to_lab(rgb_color):
    """RGB色をLAB色空間に変換"""
    # RGB値を0-1の範囲に正規化
    rgb_normalized = np.array(rgb_color).reshape(1, 1, 3) / 255.0
    # LAB色空間に変換
    lab = skcolor.rgb2lab(rgb_normalized)
    return lab[0, 0]


def ciede2000_distance(lab1, lab2):
    """CIEDE2000色差を計算"""
    # skimageのdeltae_ciede2000関数を使用
    # lab1とlab2は(L, a, b)の形式
    lab1_reshaped = np.array(lab1).reshape(1, 1, 3)
    lab2_reshaped = np.array(lab2).reshape(1, 1, 3)
    
    # CIEDE2000色差を計算
    delta_e = skcolor.deltaE_ciede2000(lab1_reshaped, lab2_reshaped)
    return delta_e[0, 0]


def find_closest_cluster_color(pixel_color, cluster_colors):
    """
    ピクセルの色に最も近いクラスタ色を見つける
    
    Args:
        pixel_color: RGB形式のピクセル色 (R, G, B)
        cluster_colors: クラスタID -> RGB色の辞書
    
    Returns:
        最も近いクラスタのRGB色
    """
    if not cluster_colors:
        return pixel_color
    
    # ピクセル色をLAB色空間に変換
    pixel_lab = rgb_to_lab(pixel_color)
    
    min_distance = float('inf')
    closest_color = pixel_color
    
    # 各クラスタ色との距離を計算
    for cluster_id, cluster_color in cluster_colors.items():
        cluster_lab = rgb_to_lab(cluster_color)
        distance = ciede2000_distance(pixel_lab, cluster_lab)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = cluster_color
    
    return closest_color


def process_fill_space_with_clusters(binary_image, original_image, cluster_info, invert_binary=True):
    """
    線画の下のピクセルをクラスタ色で塗りつぶす
    
    Args:
        binary_image: バイナリ画像（線画）
        original_image: 元の塗り画像（3枚目の入力）
        cluster_info: Fill Area Simpleからのクラスタ情報
        invert_binary: バイナリ画像を反転するか
    
    Returns:
        処理済みの画像
    """
    # バイナリ画像をグレースケールに変換
    if binary_image.mode != 'L':
        binary_gray = binary_image.convert('L')
    else:
        binary_gray = binary_image
    
    # 必要に応じてバイナリ画像を反転
    if invert_binary:
        binary_gray = ImageOps.invert(binary_gray)
    
    binary_array = np.array(binary_gray)
    original_array = np.array(original_image.convert('RGB'))
    
    # クラスタ色情報を取得
    cluster_colors = cluster_info.get('colors', {})
    
    # 出力画像を初期化（元画像のコピー）
    output_array = original_array.copy()
    
    # 白ピクセル（線画の下）の座標を取得
    white_pixels = np.argwhere(binary_array == 255)
    
    print(f"[FillSpaceV2] Processing {len(white_pixels)} pixels under line art")
    print(f"[FillSpaceV2] Using {len(cluster_colors)} color clusters")
    
    # 各白ピクセルに対して処理
    for y, x in white_pixels:
        # 元画像のピクセル色を取得
        original_color = tuple(original_array[y, x])
        
        # 最も近いクラスタ色を見つける
        closest_color = find_closest_cluster_color(original_color, cluster_colors)
        
        # ピクセルを最も近いクラスタ色で塗りつぶす
        output_array[y, x] = closest_color
    
    return Image.fromarray(output_array.astype(np.uint8))


class FillSpaceV2Node:
    """
    線画の下のピクセルをクラスタ色で塗りつぶすノード
    CIEDE2000色差を使用して最も近いクラスタ色を選択
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),  # 線画（1枚目）
                "flat_image": ("IMAGE",),    # バケツ塗りされた画像（2枚目）
                "original_image": ("IMAGE",), # 元の塗り画像（3枚目）
                "cluster_info": ("CLUSTER_INFO",),  # Fill Area Simpleからのクラスタ情報
                "invert_binary": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Invert Binary Image"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "preview")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, flat_image, original_image, cluster_info, invert_binary=True):
        """
        線画の下のピクセルをクラスタ色で塗りつぶす処理
        
        Args:
            binary_image: バイナリ画像（線画）のテンソル
            flat_image: バケツ塗りされた画像のテンソル
            original_image: 元の塗り画像のテンソル
            cluster_info: クラスタ情報の辞書
            invert_binary: バイナリ画像を反転するか
        
        Returns:
            filled_image: 処理済みの画像
            preview: 処理前後の比較画像
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        flat_pil = tensor_to_pil(flat_image)
        original_pil = tensor_to_pil(original_image)
        
        # 線画の下のピクセルを処理
        filled_image = process_fill_space_with_clusters(
            binary_pil, 
            original_pil,
            cluster_info,
            invert_binary
        )
        
        # プレビュー画像の作成（前後比較）
        preview = create_before_after_preview(original_pil, filled_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(filled_image)
        preview_tensor = pil_to_tensor(preview)
        
        return (output_tensor, preview_tensor)


def create_before_after_preview(original, processed):
    """処理前後の比較画像を作成"""
    width = original.width
    height = original.height
    
    # 左右に並べた比較画像を作成
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original.convert('RGB'), (0, 0))
    comparison.paste(processed.convert('RGB'), (width, 0))
    
    # 中央に区切り線を追加
    from PIL import ImageDraw
    draw = ImageDraw.Draw(comparison)
    draw.line([(width, 0), (width, height)], fill=(255, 0, 0), width=2)
    
    # ラベルを追加
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((width + 10, 10), "Processed", fill=(255, 255, 255))
    
    return comparison


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillSpaceV2": FillSpaceV2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillSpaceV2": "Fill Space V2 (Cluster-based)",
}
