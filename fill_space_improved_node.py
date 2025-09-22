"""
Improved Fill Space Node for ComfyUI
線画の隙間を高精度で埋める改良版ノード
アンチエイリアシングやグレースケール値に対応
"""

import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
from collections import deque
from scipy import ndimage
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


def preprocess_lineart(lineart_image, threshold=200, dilate_size=1, smooth_edges=True):
    """
    線画の前処理
    アンチエイリアシングの処理と線画領域の明確化
    
    Args:
        lineart_image: 線画画像（PIL Image）
        threshold: 線画と判定する閾値
        dilate_size: 線画を太くするサイズ
        smooth_edges: エッジのスムージング
    
    Returns:
        processed_mask: 前処理済みマスク（線画=0, 塗りつぶし領域=255）
    """
    # グレースケール変換
    if lineart_image.mode != 'L':
        lineart_gray = lineart_image.convert('L')
    else:
        lineart_gray = lineart_image.copy()
    
    lineart_array = np.array(lineart_gray)
    
    # アンチエイリアシング対策：エッジ検出による線画領域の特定
    edges = cv2.Canny(lineart_array, 50, 150)
    
    # 閾値処理（線画を黒、背景を白とする）
    binary_mask = np.where(lineart_array < threshold, 0, 255).astype(np.uint8)
    
    # エッジ部分を確実に線画として扱う
    binary_mask[edges > 0] = 0
    
    # モルフォロジー処理で線画を整理
    if dilate_size > 0:
        kernel = np.ones((dilate_size*2+1, dilate_size*2+1), np.uint8)
        # 線画を太くする（白い領域を縮小）
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
    if smooth_edges:
        # ガウシアンブラーでエッジを滑らかに
        binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)
        # 再度二値化
        binary_mask = np.where(binary_mask < 128, 0, 255).astype(np.uint8)
    
    return binary_mask


def create_confidence_map(binary_mask, lineart_original):
    """
    信頼度マップの作成
    線画からの距離と元の輝度値に基づいて、塗りつぶしの信頼度を計算
    
    Args:
        binary_mask: 二値化マスク
        lineart_original: 元の線画
    
    Returns:
        confidence_map: 信頼度マップ（0-1）
    """
    # 距離変換
    distance_map = cv2.distanceTransform(
        binary_mask, 
        cv2.DIST_L2, 
        cv2.DIST_MASK_PRECISE
    )
    
    # 正規化
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    
    # 元の線画の輝度値も考慮
    if lineart_original.mode != 'L':
        lineart_gray = lineart_original.convert('L')
    else:
        lineart_gray = lineart_original
    
    lineart_array = np.array(lineart_gray).astype(np.float32) / 255.0
    
    # 白い部分（塗りつぶし対象）ほど信頼度が高い
    brightness_confidence = lineart_array
    
    # 距離と輝度の組み合わせ
    confidence_map = distance_map * brightness_confidence
    
    return confidence_map


def improved_fill_space(mask_image, source_image, confidence_map=None, 
                       interpolation_mode='weighted', blur_size=0):
    """
    改良された隙間埋め処理
    
    Args:
        mask_image: 処理済みマスク画像（0=線画, 255=塗りつぶし領域）
        source_image: 元画像
        confidence_map: 信頼度マップ
        interpolation_mode: 補間モード（'nearest', 'weighted', 'inpaint'）
        blur_size: 後処理のブラーサイズ
    
    Returns:
        filled_image: 塗りつぶし済み画像
    """
    mask_array = np.array(mask_image)
    source_array = np.array(source_image)
    h, w = mask_array.shape[:2]
    
    if interpolation_mode == 'inpaint':
        # OpenCVのインペインティングを使用（最も自然な結果）
        inpaint_mask = (mask_array == 255).astype(np.uint8) * 255
        
        # Navier-Stokes方程式ベースのインペインティング
        result = cv2.inpaint(
            source_array, 
            inpaint_mask, 
            inpaintRadius=3,
            flags=cv2.INPAINT_NS
        )
        
    elif interpolation_mode == 'weighted':
        # 距離加重補間
        result = weighted_fill(mask_array, source_array, confidence_map)
        
    else:  # 'nearest'
        # 最近傍補間（高速だが品質は劣る）
        result = nearest_neighbor_fill(mask_array, source_array)
    
    # 後処理：境界部分のブラー
    if blur_size > 0:
        # 境界部分のみブラーを適用
        boundary = find_boundary_pixels(mask_array)
        if len(boundary) > 0:
            blurred = cv2.GaussianBlur(result, (blur_size*2+1, blur_size*2+1), 0)
            for y, x in boundary:
                # 境界付近のピクセルのみブレンド
                alpha = 0.5
                result[y, x] = (1-alpha) * result[y, x] + alpha * blurred[y, x]
    
    return Image.fromarray(result.astype(np.uint8))


def weighted_fill(mask_array, source_array, confidence_map=None):
    """
    距離加重補間による塗りつぶし
    複数の最近傍点からの重み付け平均を使用
    """
    h, w = mask_array.shape[:2]
    result = source_array.copy()
    
    # 塗りつぶし対象のピクセル
    fill_pixels = np.argwhere(mask_array == 255)
    
    # 線画（ソース）のピクセル
    source_pixels = np.argwhere(mask_array == 0)
    
    if len(source_pixels) == 0 or len(fill_pixels) == 0:
        return result
    
    # KD-Treeを使用して効率的に最近傍を検索
    from scipy.spatial import cKDTree
    tree = cKDTree(source_pixels)
    
    # 各塗りつぶしピクセルに対して処理
    for fy, fx in fill_pixels:
        # 最近傍k個の点を取得
        k = min(8, len(source_pixels))
        distances, indices = tree.query([fy, fx], k=k)
        
        # 距離が0の場合の処理
        if distances[0] == 0:
            nearest_y, nearest_x = source_pixels[indices[0]]
            result[fy, fx] = source_array[nearest_y, nearest_x]
            continue
        
        # 距離に基づく重み計算（逆数）
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum()
        
        # 信頼度マップがある場合は考慮
        if confidence_map is not None:
            conf_weights = []
            for idx in indices:
                sy, sx = source_pixels[idx]
                conf_weights.append(1.0 - confidence_map[sy, sx])
            conf_weights = np.array(conf_weights)
            weights = weights * conf_weights
            weights = weights / (weights.sum() + 1e-6)
        
        # 重み付け平均で色を計算
        weighted_color = np.zeros(source_array.shape[2] if len(source_array.shape) > 2 else 1)
        for weight, idx in zip(weights, indices):
            sy, sx = source_pixels[idx]
            if len(source_array.shape) > 2:
                weighted_color += weight * source_array[sy, sx]
            else:
                weighted_color += weight * source_array[sy, sx]
        
        result[fy, fx] = weighted_color.astype(np.uint8)
    
    return result


def nearest_neighbor_fill(mask_array, source_array):
    """
    最近傍補間による塗りつぶし（従来の方法の改良版）
    """
    h, w = mask_array.shape[:2]
    
    # 8方向で探索
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                  (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # 距離マップと最近傍マップの初期化
    distance_map = np.full((h, w), np.inf)
    nearest_map = np.zeros((h, w, 2), dtype=np.int32)
    
    # 線画ピクセルから開始
    queue = deque()
    source_pixels = np.argwhere(mask_array == 0)
    
    for y, x in source_pixels:
        distance_map[y, x] = 0
        nearest_map[y, x] = [y, x]
        queue.append((y, x))
    
    # BFSで距離マップを構築
    while queue:
        cy, cx = queue.popleft()
        current_dist = distance_map[cy, cx]
        
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                # 斜め移動の場合は距離を√2倍
                new_dist = current_dist + (1.414 if dy != 0 and dx != 0 else 1)
                
                if new_dist < distance_map[ny, nx]:
                    distance_map[ny, nx] = new_dist
                    nearest_map[ny, nx] = nearest_map[cy, cx]
                    queue.append((ny, nx))
    
    # 結果画像の作成
    result = source_array.copy()
    fill_pixels = np.argwhere(mask_array == 255)
    
    for y, x in fill_pixels:
        nearest_y, nearest_x = nearest_map[y, x].astype(int)
        result[y, x] = source_array[nearest_y, nearest_x]
    
    return result


def find_boundary_pixels(mask_array, width=2):
    """
    マスクの境界ピクセルを検出
    """
    # エッジ検出
    edges = cv2.Canny(mask_array, 50, 150)
    
    # 境界を少し広げる
    if width > 1:
        kernel = np.ones((width*2+1, width*2+1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    boundary_pixels = np.argwhere(edges > 0)
    return boundary_pixels


class FillSpaceImprovedNode:
    """
    改良版隙間埋めノード
    アンチエイリアシングと中間値に対応
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lineart_image": ("IMAGE",),
                "flat_image": ("IMAGE",),
                "line_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Line Detection Threshold"
                }),
                "interpolation_mode": (["weighted", "nearest", "inpaint"],),
                "preprocess_mode": (["adaptive", "simple", "advanced"],),
                "line_dilate": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Line Dilation"
                }),
                "smooth_edges": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Smooth Edges"
                }),
                "post_blur": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Post-process Blur"
                }),
                "use_confidence": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Use Confidence Map"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("filled_image", "comparison", "processed_mask", "confidence_visualization")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, lineart_image, flat_image, line_threshold=200, 
                interpolation_mode="weighted", preprocess_mode="adaptive",
                line_dilate=1, smooth_edges=True, post_blur=1, use_confidence=True):
        """
        改良版隙間埋め処理
        """
        
        # テンソルをPIL Imageに変換
        lineart_pil = tensor_to_pil(lineart_image)
        flat_pil = tensor_to_pil(flat_image)
        
        # 前処理モードの選択
        if preprocess_mode == "advanced":
            # より高度な前処理（エッジ検出+モルフォロジー）
            processed_mask = preprocess_lineart(
                lineart_pil, 
                threshold=line_threshold,
                dilate_size=line_dilate,
                smooth_edges=smooth_edges
            )
        elif preprocess_mode == "adaptive":
            # 適応的閾値処理
            if lineart_pil.mode != 'L':
                lineart_gray = lineart_pil.convert('L')
            else:
                lineart_gray = lineart_pil
            
            lineart_array = np.array(lineart_gray)
            
            # 適応的閾値処理
            adaptive_thresh = cv2.adaptiveThreshold(
                lineart_array, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            
            # モルフォロジー処理
            if line_dilate > 0:
                kernel = np.ones((line_dilate*2+1, line_dilate*2+1), np.uint8)
                processed_mask = cv2.erode(adaptive_thresh, kernel, iterations=1)
            else:
                processed_mask = adaptive_thresh
                
        else:  # simple
            # シンプルな閾値処理
            if lineart_pil.mode != 'L':
                lineart_gray = lineart_pil.convert('L')
            else:
                lineart_gray = lineart_pil
            
            lineart_array = np.array(lineart_gray)
            processed_mask = np.where(lineart_array < line_threshold, 0, 255).astype(np.uint8)
        
        # 信頼度マップの作成
        confidence_map = None
        if use_confidence and interpolation_mode == "weighted":
            confidence_map = create_confidence_map(processed_mask, lineart_pil)
        
        # 隙間埋め処理
        filled_image = improved_fill_space(
            processed_mask,
            flat_pil,
            confidence_map=confidence_map,
            interpolation_mode=interpolation_mode,
            blur_size=post_blur
        )
        
        # 比較画像の作成
        comparison = create_comparison_image(flat_pil, filled_image, processed_mask)
        
        # 信頼度マップの可視化
        if confidence_map is not None:
            confidence_vis = visualize_confidence_map(confidence_map)
        else:
            confidence_vis = Image.new('RGB', (flat_pil.width, flat_pil.height), (128, 128, 128))
        
        # ComfyUIのテンソル形式に変換
        filled_tensor = pil_to_tensor(filled_image)
        comparison_tensor = pil_to_tensor(comparison)
        confidence_tensor = pil_to_tensor(confidence_vis)
        
        # マスクをテンソルに変換
        mask_array = processed_mask.astype(np.float32) / 255.0
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_tensor = torch.from_numpy(mask_array)
        
        return (filled_tensor, comparison_tensor, mask_tensor, confidence_tensor)


def create_comparison_image(original, filled, mask):
    """
    オリジナル、マスク、結果の比較画像を作成
    """
    width = original.width
    height = original.height
    
    # 3つの画像を横に並べる
    comparison = Image.new('RGB', (width * 3, height))
    
    # オリジナル
    comparison.paste(original.convert('RGB'), (0, 0))
    
    # マスク
    mask_rgb = Image.fromarray(mask).convert('RGB')
    comparison.paste(mask_rgb, (width, 0))
    
    # 結果
    comparison.paste(filled.convert('RGB'), (width * 2, 0))
    
    # ラベルを追加
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    try:
        # フォントの取得を試みる
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # ラベル背景
    label_height = 30
    draw.rectangle([(0, 0), (width * 3, label_height)], fill=(0, 0, 0, 180))
    
    # テキスト
    draw.text((10, 5), "Original", fill=(255, 255, 255), font=font)
    draw.text((width + 10, 5), "Processed Mask", fill=(255, 255, 255), font=font)
    draw.text((width * 2 + 10, 5), "Result", fill=(255, 255, 255), font=font)
    
    return comparison


def visualize_confidence_map(confidence_map):
    """
    信頼度マップをカラーマップで可視化
    """
    # 0-255にスケール
    conf_uint8 = (confidence_map * 255).astype(np.uint8)
    
    # カラーマップ適用
    colored = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(colored)


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillSpaceImproved": FillSpaceImprovedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillSpaceImproved": "Fill Space (Improved)",
}
