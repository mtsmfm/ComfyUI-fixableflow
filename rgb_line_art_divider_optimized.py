"""
RGB Line Art Layer Divider - OPTIMIZED VERSION
高速化版：KMeansクラスタリングとベクトル化処理を使用
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_convertor import pil2cv
from pytoshop.enums import BlendMode
import cv2
from sklearn.cluster import KMeans
from scipy import ndimage
import pytoshop
from pytoshop.core import PsdFile
from pytoshop.user import nested_layers
from pytoshop import enums
from datetime import datetime

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


def extract_color_regions_optimized(base_image_cv, tolerance=10, use_kmeans=True, max_colors=50):
    """
    最適化版：下塗り画像からRGB値ごとに領域を抽出
    まず同じRGB値でグループ化し、その後グループ単位でクラスタリング
    
    Args:
        base_image_cv: 下塗り画像（BGRA形式）
        tolerance: 同じ色と判定する許容値
        use_kmeans: KMeansクラスタリングを使用するか
        max_colors: 最大色数（KMeans使用時）
    
    Returns:
        color_regions: {(R,G,B): mask} の辞書
    """
    # BGRAからRGBに変換
    if base_image_cv.shape[2] == 4:
        bgr_image = base_image_cv[:, :, :3]
        alpha = base_image_cv[:, :, 3]
    else:
        bgr_image = base_image_cv
        alpha = np.ones((base_image_cv.shape[0], base_image_cv.shape[1]), dtype=np.uint8) * 255
    
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    
    # アルファマスクを作成
    valid_mask = alpha > 0
    
    # ステップ1: 同じRGB値のピクセルをグループ化
    print("Step 1: Grouping pixels by unique colors...")
    
    # 有効なピクセルのみを取得
    valid_pixels = rgb_image[valid_mask]
    
    # ユニークな色とその出現回数、インデックスを効率的に取得
    # reshape to 2D array for unique operation
    pixel_array = rgb_image.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    
    # Get unique colors with inverse indices
    unique_colors, inverse_indices = np.unique(
        pixel_array[valid_flat], 
        axis=0, 
        return_inverse=True
    )
    
    print(f"Found {len(unique_colors)} unique colors")
    
    # 各ユニーク色のピクセル数を計算
    color_counts = np.bincount(inverse_indices)
    
    # 色グループの情報を作成
    color_groups = []
    for i, (color, count) in enumerate(zip(unique_colors, color_counts)):
        if count > 0:
            color_groups.append({
                'color': color,
                'count': count,
                'index': i
            })
    
    # ステップ2: 近いRGB値のグループを統合
    if use_kmeans and len(color_groups) > 1:
        print(f"Step 2: Clustering {len(color_groups)} color groups...")
        
        # 色とその重み（ピクセル数）を準備
        colors = np.array([g['color'] for g in color_groups])
        weights = np.array([g['count'] for g in color_groups])
        
        # クラスタ数を決定
        n_clusters = min(len(colors), max_colors)
        
        if n_clusters < len(colors):
            # 重み付きKMeansクラスタリング
            # 各色を重み回数分複製するのではなく、sample_weightを使用
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
            
            # 重みを考慮したクラスタリング
            cluster_labels = kmeans.fit_predict(colors, sample_weight=weights)
            
            # クラスタ中心を計算（重み付き平均）
            cluster_centers = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_colors = colors[cluster_mask]
                    cluster_weights = weights[cluster_mask]
                    # 重み付き平均でクラスタ中心を計算
                    weighted_center = np.average(cluster_colors, axis=0, weights=cluster_weights)
                    cluster_centers.append(weighted_center.astype(int))
            
            # 元の色インデックスをクラスタラベルにマップ
            color_to_cluster = {}
            for i, group in enumerate(color_groups):
                color_to_cluster[group['index']] = cluster_labels[i]
            
            # クラスタごとのマスクを作成
            color_regions = {}
            for cluster_id, center_color in enumerate(cluster_centers):
                # このクラスタに属する全ピクセルのマスクを作成
                cluster_mask = np.zeros((height, width), dtype=np.uint8)
                
                # inverse_indicesを使用して効率的にマスクを作成
                for color_idx, group in enumerate(color_groups):
                    if cluster_labels[color_idx] == cluster_id:
                        # この色グループに属するピクセルを取得
                        group_pixel_mask = (inverse_indices == group['index'])
                        # フルサイズのマスクに変換
                        full_mask = np.zeros(len(valid_flat), dtype=bool)
                        full_mask[valid_flat] = group_pixel_mask
                        cluster_mask += full_mask.reshape(height, width).astype(np.uint8) * 255
                
                if np.any(cluster_mask):
                    color_regions[tuple(center_color)] = cluster_mask
        else:
            # クラスタリング不要（色数が既に少ない）
            color_regions = create_masks_from_groups(
                rgb_image, valid_mask, unique_colors, inverse_indices, height, width
            )
    
    elif tolerance > 0:
        print(f"Step 2: Merging colors within tolerance {tolerance}...")
        
        # 許容値ベースのグループ統合
        merged_groups = merge_similar_colors(color_groups, tolerance)
        
        # マスクを作成
        color_regions = {}
        for merged_group in merged_groups:
            mask = np.zeros((height, width), dtype=np.uint8)
            for color_idx in merged_group['indices']:
                group_pixel_mask = (inverse_indices == color_idx)
                full_mask = np.zeros(len(valid_flat), dtype=bool)
                full_mask[valid_flat] = group_pixel_mask
                mask += full_mask.reshape(height, width).astype(np.uint8) * 255
            
            if np.any(mask):
                color_regions[tuple(merged_group['center'])] = mask
    
    else:
        # グループ化なし
        color_regions = create_masks_from_groups(
            rgb_image, valid_mask, unique_colors, inverse_indices, height, width
        )
    
    print(f"Final: {len(color_regions)} color regions")
    return color_regions


def create_masks_from_groups(rgb_image, valid_mask, unique_colors, inverse_indices, height, width):
    """
    ユニーク色グループからマスクを作成
    """
    color_regions = {}
    valid_flat = valid_mask.reshape(-1)
    
    for i, color in enumerate(unique_colors):
        # この色に属するピクセルのマスクを作成
        group_pixel_mask = (inverse_indices == i)
        full_mask = np.zeros(len(valid_flat), dtype=bool)
        full_mask[valid_flat] = group_pixel_mask
        mask = full_mask.reshape(height, width).astype(np.uint8) * 255
        
        if np.any(mask):
            color_regions[tuple(color)] = mask
    
    return color_regions


def merge_similar_colors(color_groups, tolerance):
    """
    許容値内の類似色をマージ
    """
    if len(color_groups) == 0:
        return []
    
    # 色グループをピクセル数でソート（大きいグループを優先）
    sorted_groups = sorted(color_groups, key=lambda x: x['count'], reverse=True)
    
    merged = []
    used = set()
    
    for i, group in enumerate(sorted_groups):
        if i in used:
            continue
        
        # 新しいマージグループを開始
        merge_group = {
            'indices': [group['index']],
            'colors': [group['color']],
            'total_count': group['count']
        }
        used.add(i)
        
        # 類似色を探す
        for j, other_group in enumerate(sorted_groups[i+1:], i+1):
            if j in used:
                continue
            
            # 色差を計算
            color_diff = np.abs(group['color'] - other_group['color'])
            if np.all(color_diff <= tolerance):
                merge_group['indices'].append(other_group['index'])
                merge_group['colors'].append(other_group['color'])
                merge_group['total_count'] += other_group['count']
                used.add(j)
        
        # マージグループの中心色を計算（重み付き平均）
        colors = np.array(merge_group['colors'])
        weights = np.array([g['count'] for g in sorted_groups if g['index'] in merge_group['indices']])
        center_color = np.average(colors, axis=0, weights=weights).astype(int)
        merge_group['center'] = center_color
        
        merged.append(merge_group)
    
    return merged


def merge_small_regions_optimized(color_regions, min_region_size=100):
    """
    最適化版：小さい領域を効率的にマージ
    """
    filtered_regions = {}
    small_regions = []
    
    for color, mask in color_regions.items():
        # 連結成分分析を使用して実際の領域サイズを計算
        num_labels, labels = cv2.connectedComponents(mask)
        
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id).astype(np.uint8) * 255
            component_size = np.sum(component_mask > 0)
            
            if component_size >= min_region_size:
                # 既存の同色マスクに追加
                if color in filtered_regions:
                    filtered_regions[color] = cv2.bitwise_or(
                        filtered_regions[color], component_mask
                    )
                else:
                    filtered_regions[color] = component_mask
            else:
                small_regions.append(component_mask)
    
    # 小さい領域を「その他」として統合
    if small_regions:
        combined_small = np.zeros_like(small_regions[0])
        for mask in small_regions:
            combined_small = cv2.bitwise_or(combined_small, mask)
        if np.any(combined_small > 0):
            filtered_regions[(128, 128, 128)] = combined_small
    
    return filtered_regions


def create_region_layers(base_image_cv, color_regions):
    """
    色領域ごとにレイヤーを作成（変更なし）
    """
    layers = []
    names = []
    
    for color, mask in color_regions.items():
        # マスクを適用してレイヤーを作成
        layer = np.zeros_like(base_image_cv)
        
        # ベクトル化処理で高速化
        mask_bool = mask > 0
        layer[mask_bool] = base_image_cv[mask_bool]
        
        layers.append(layer)
        names.append(f"Color_R{color[0]}_G{color[1]}_B{color[2]}")
    
    return layers, names


def save_psd_with_nested_layers(base_image_cv, line_art_cv, color_layers, layer_names, 
                                output_dir, blend_mode=BlendMode.multiply, filename_prefix="rgb_divided"):
    """
    nested_layersを使用してPSDファイルを保存（変更なし）
    """
    # ファイル名生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.psd")
    
    height, width = base_image_cv.shape[:2]
    layers_list = []
    
    # 背景レイヤーを追加
    bg_arr = base_image_cv[:, :, [2, 1, 0]]  # BGRからRGBに変換
    if base_image_cv.shape[2] == 4:
        channels = [bg_arr[:, :, 0], bg_arr[:, :, 1], bg_arr[:, :, 2], base_image_cv[:, :, 3]]
    else:
        channels = [bg_arr[:, :, 0], bg_arr[:, :, 1], bg_arr[:, :, 2]]
    
    bg_layer = nested_layers.Image(
        name="Background",
        visible=True,
        opacity=255,
        group_id=0,
        blend_mode=enums.BlendMode.normal,
        top=0,
        left=0,
        channels=channels,
        metadata=None,
        layer_color=0,
        color_mode=None
    )
    layers_list.append(bg_layer)
    
    # 色領域レイヤーを追加
    for layer_data, name in zip(color_layers, layer_names):
        rgb_data = layer_data[:, :, [2, 1, 0]]  # BGRからRGBに変換
        if layer_data.shape[2] == 4:
            channels = [rgb_data[:, :, 0], rgb_data[:, :, 1], rgb_data[:, :, 2], layer_data[:, :, 3]]
        else:
            channels = [rgb_data[:, :, 0], rgb_data[:, :, 1], rgb_data[:, :, 2]]
        
        layer = nested_layers.Image(
            name=name,
            visible=True,
            opacity=255,
            group_id=0,
            blend_mode=enums.BlendMode.normal,
            top=0,
            left=0,
            channels=channels,
            metadata=None,
            layer_color=0,
            color_mode=None
        )
        layers_list.append(layer)
    
    # 線画レイヤーを最上位に追加
    line_rgb = line_art_cv[:, :, [2, 1, 0]]  # BGRからRGBに変換
    if line_art_cv.shape[2] == 4:
        channels = [line_rgb[:, :, 0], line_rgb[:, :, 1], line_rgb[:, :, 2], line_art_cv[:, :, 3]]
    else:
        channels = [line_rgb[:, :, 0], line_rgb[:, :, 1], line_rgb[:, :, 2]]
    
    line_layer = nested_layers.Image(
        name="Line Art",
        visible=True,
        opacity=255,
        group_id=0,
        blend_mode=blend_mode,
        top=0,
        left=0,
        channels=channels,
        metadata=None,
        layer_color=0,
        color_mode=None
    )
    layers_list.append(line_layer)
    
    # PSDファイルとして保存
    output = nested_layers.nested_layers_to_psd(layers_list, color_mode=3)  # RGB mode
    with open(filename, 'wb') as f:
        output.write(f)
    
    return filename


class RGBLineArtDividerOptimized:
    """
    最適化版：RGB線画と下塗り画像から領域分割PSDを生成するノード
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
                "use_kmeans": ("BOOLEAN", {
                    "default": True,
                    "label": "Use KMeans Clustering (faster)"
                }),
                "max_colors": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "display": "slider",
                    "label": "Maximum Colors (KMeans only)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("composite", "base_color", "layer_count", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, line_art, base_color, color_tolerance, line_blend_mode, 
                merge_small_regions, min_region_size, use_kmeans, max_colors):
        
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
        
        # 色領域を抽出（最適化版を使用）
        print(f"Extracting color regions (KMeans: {use_kmeans})...")
        color_regions = extract_color_regions_optimized(
            base_color_cv, 
            tolerance=color_tolerance,
            use_kmeans=use_kmeans,
            max_colors=max_colors
        )
        print(f"Found {len(color_regions)} color regions")
        
        # 小さい領域をマージ（最適化版）
        if merge_small_regions:
            print("Merging small regions...")
            color_regions = merge_small_regions_optimized(color_regions, min_region_size)
            print(f"After merging: {len(color_regions)} regions")
        
        # レイヤーを作成
        color_layers, layer_names = create_region_layers(base_color_cv, color_regions)
        
        # BlendModeの設定
        blend_mode_map = {
            "multiply": enums.BlendMode.multiply,
            "normal": enums.BlendMode.normal,
            "darken": enums.BlendMode.darken,
            "overlay": enums.BlendMode.overlay
        }
        
        # PSDファイルを保存
        filename = save_psd_with_nested_layers(
            base_color_cv,
            line_art_cv,
            color_layers,
            layer_names,
            output_dir,
            blend_mode_map[line_blend_mode],
            "rgb_divided_optimized"
        )
        
        print(f"PSD file saved: {filename}")
        print(f"Created {len(color_regions)} color region layers")
        
        # コンポジット画像を作成（プレビュー用）
        composite = base_color_cv.copy()
        if line_blend_mode == "multiply":
            # 乗算合成（ベクトル化）
            line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
            composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
            composite[:, :, :3] = (composite_rgb * line_rgb * 255).astype(np.uint8)
        elif line_blend_mode == "normal":
            # アルファブレンディング（ベクトル化）
            if line_art_cv.shape[2] == 4:
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
