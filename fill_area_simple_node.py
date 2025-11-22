"""
Simple Fill Area Node for ComfyUI
塗り領域を単純に均一化するノード（統合処理なし）
SplitAreaNodeやカラーラベル画像からのregion_mask入力を使用
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
from sklearn.cluster import KMeans
from skimage import color as skcolor
import folder_paths
import os

# パス設定
comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'
output_dir = f"{layer_divider_path}/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def tensor_to_pil(tensor):
    """ComfyUIのテンソル形式をPIL Imageに変換"""
    image_np = tensor.cpu().detach().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # バッチの最初の画像を取得
    image_np = (image_np * 255).astype(np.uint8)

    if image_np.ndim == 2:
        mode = "L"
        return Image.fromarray(image_np, mode=mode)

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
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1)  # (H, W) -> (H, W, 1)
    image_np = np.expand_dims(image_np, axis=0)       # (H, W, C) -> (1, H, W, C)
    return torch.from_numpy(image_np)


def get_most_frequent_color(image_array, mask):
    """指定された領域内の最頻出色を取得"""
    pixels = image_array[mask]
    if len(pixels) == 0:
        return (0, 0, 0)

    if len(pixels.shape) == 1:
        return tuple(pixels)
    else:
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_frequent_idx = np.argmax(counts)
        return tuple(unique_colors[most_frequent_idx])


def convert_mask_to_labels(mask_np):
    """
    region_mask画像(numpy)をラベル配列(H, W)に変換する。
    - RGBラベル画像（例：各パーツごとに別の色）にも対応
    - グレースケール/float(0〜1)マスクにも対応
    - 最も頻度の高い値/色を背景(label=0)として扱い、それ以外を1..Nのラベルに割り当てる
    """
    # 4次元は想定しないが、念のため (B, H, W, C) -> (H, W, C)
    if mask_np.ndim == 4:
        mask_np = mask_np[0]

    # チャンネル1枚の3次元は (H, W)
    if mask_np.ndim == 3 and mask_np.shape[2] == 1:
        mask_np = mask_np[:, :, 0]

    # float の場合は 0〜255 に量子化して uint8 に
    if np.issubdtype(mask_np.dtype, np.floating):
        mask_np = np.clip(np.round(mask_np * 255.0), 0, 255).astype(np.uint8)

    # --- カラーラベル (H, W, 3 or 4) ---
    if mask_np.ndim == 3:
        # RGBA の場合も RGB 部分だけ使う
        color_np = mask_np[:, :, :3].reshape(-1, 3)  # (H*W, 3)
        unique_colors, inverse = np.unique(color_np, axis=0, return_inverse=True)
        counts = np.bincount(inverse)

        if len(unique_colors) == 0:
            labeled_array = np.zeros(mask_np.shape[:2], dtype=np.int32)
            return labeled_array, 0, unique_colors

        # 一番頻度の高い色を背景として0にする
        bg_idx = int(np.argmax(counts))

        labeled_flat = np.zeros_like(inverse, dtype=np.int32)
        next_label = 1
        for idx in range(len(unique_colors)):
            if idx == bg_idx:
                continue
            labeled_flat[inverse == idx] = next_label
            next_label += 1

        labeled_array = labeled_flat.reshape(mask_np.shape[:2])
        num_features = next_label - 1
        return labeled_array, num_features, unique_colors

    # --- グレースケールラベル (H, W) ---
    elif mask_np.ndim == 2:
        vals = mask_np.reshape(-1)
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        counts = np.bincount(inverse)

        if len(unique_vals) == 0:
            labeled_array = np.zeros(mask_np.shape, dtype=np.int32)
            return labeled_array, 0, unique_vals

        bg_idx = int(np.argmax(counts))

        labeled_flat = np.zeros_like(inverse, dtype=np.int32)
        next_label = 1
        for idx in range(len(unique_vals)):
            if idx == bg_idx:
                continue
            labeled_flat[inverse == idx] = next_label
            next_label += 1

        labeled_array = labeled_flat.reshape(mask_np.shape[:2])
        num_features = next_label - 1
        return labeled_array, num_features, unique_vals

    else:
        # 想定外の形状は全部背景扱い
        labeled_array = np.zeros(mask_np.shape[:2], dtype=np.int32)
        return labeled_array, 0, None


def merge_similar_regions_by_color_distance(image, labeled_array, num_features, distance_threshold=10):
    """
    CIEDE2000色距離を使用して類似した色の領域を統合

    Args:
        image: PIL Image
        labeled_array: ラベル配列
        num_features: 現在の領域数
        distance_threshold: CIEDE2000距離の閾値（デフォルト10）

    Returns:
        新しいラベル配列, 新しい領域数
    """
    image_array = np.array(image)

    # 各領域の代表色（最頻出色）とピクセル数を取得
    region_info = []

    for label_id in range(1, num_features + 1):
        mask = labeled_array == label_id
        if np.any(mask):
            # 領域の最頻出色を取得
            most_frequent_color = get_most_frequent_color(image_array, mask)
            # LAB色空間に変換
            lab_color = skcolor.rgb2lab(np.array(most_frequent_color).reshape(1, 1, 3) / 255.0)[0, 0]
            pixel_count = np.sum(mask)
            region_info.append({
                'id': label_id,
                'color': most_frequent_color,
                'lab': lab_color,
                'pixel_count': pixel_count,
                'merged': False
            })

    if len(region_info) == 0:
        return labeled_array, 0

    # ピクセル数で降順ソート
    region_info.sort(key=lambda x: x['pixel_count'], reverse=True)

    # マージグループを作成
    merge_groups = []
    processed_regions = set()

    for main_region in region_info:
        if main_region['id'] in processed_regions:
            continue

        # 新しいグループを開始
        current_group = [main_region['id']]
        processed_regions.add(main_region['id'])

        # 他の領域と比較
        for other_region in region_info:
            if other_region['id'] in processed_regions:
                continue

            # CIEDE2000距離を計算
            delta_e = skcolor.deltaE_ciede2000(
                main_region['lab'].reshape(1, 1, 3),
                other_region['lab'].reshape(1, 1, 3)
            )[0, 0]

            if delta_e <= distance_threshold:
                current_group.append(other_region['id']]
                processed_regions.add(other_region['id'])

        merge_groups.append(current_group)

    print(f"[FillAreaSimple] Created {len(merge_groups)} merged groups from {num_features} regions")
    print(f"[FillAreaSimple] Color distance threshold: {distance_threshold}")

    # 新しいラベル配列を作成
    new_labeled_array = np.zeros_like(labeled_array)

    # 各グループに新しいラベルを割り当て
    for new_label, group in enumerate(merge_groups, 1):
        for old_label in group:
            mask = labeled_array == old_label
            new_labeled_array[mask] = new_label

    # 背景はそのまま
    background_mask = labeled_array == 0
    new_labeled_array[background_mask] = 0

    new_num_features = len(merge_groups)

    return new_labeled_array, new_num_features


def merge_similar_regions(image, labeled_array, num_features, target_clusters):
    """
    類似した色の領域を統合してクラスタ数を削減

    Args:
        image: PIL Image
        labeled_array: ラベル配列
        num_features: 現在の領域数
        target_clusters: 目標クラスタ数

    Returns:
        新しいラベル配列, 新しい領域数
    """
    image_array = np.array(image)

    # 各領域の代表色（最頻出色）を取得
    region_colors = []
    region_ids = []

    for label_id in range(1, num_features + 1):
        mask = labeled_array == label_id
        if np.any(mask):
            # 領域の最頻出色を取得（平均色ではなく）
            most_frequent_color = get_most_frequent_color(image_array, mask)
            region_colors.append(most_frequent_color)
            region_ids.append(label_id)

    if len(region_colors) == 0:
        return labeled_array, 0

    # RGB色をLAB色空間に変換
    region_colors_array = np.array(region_colors)
    region_colors_lab = skcolor.rgb2lab(region_colors_array.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    # K-meansクラスタリングで類似色をグループ化
    kmeans = KMeans(n_clusters=min(target_clusters, len(region_colors)), random_state=42)
    cluster_labels = kmeans.fit_predict(region_colors_lab)

    # 新しいラベル配列を作成
    new_labeled_array = np.zeros_like(labeled_array)

    # 各領域を新しいクラスタIDに再マッピング
    for i, original_id in enumerate(region_ids):
        new_cluster_id = cluster_labels[i] + 1  # 0は背景用なので+1
        mask = labeled_array == original_id
        new_labeled_array[mask] = new_cluster_id

    # 背景はそのまま
    background_mask = labeled_array == 0
    new_labeled_array[background_mask] = 0

    new_num_features = len(np.unique(cluster_labels))

    return new_labeled_array, new_num_features


def fill_areas_simple(image, labeled_array, num_features):
    """各領域を最頻出色で塗りつぶす（統合処理なし）"""
    image_array = np.array(image)
    result_array = np.zeros_like(image_array)  # 新しい配列を作成

    # クラスタ色情報を保存する辞書
    cluster_colors = {}

    # デバッグ: 各領域の情報を出力
    print(f"Total regions detected: {num_features}")

    # 各ラベル領域を単一色で塗りつぶし
    for label_id in range(1, num_features + 1):
        mask = labeled_array == label_id
        if np.any(mask):
            # その領域の最頻出色を取得
            most_frequent_color = get_most_frequent_color(image_array, mask)
            print(f"Region {label_id}: color = {most_frequent_color}, pixels = {np.sum(mask)}")
            # 領域を単一色で塗りつぶし
            result_array[mask] = most_frequent_color
            # クラスタ色を保存
            cluster_colors[label_id] = most_frequent_color

    # 背景（label=0）も処理
    background_mask = labeled_array == 0
    if np.any(background_mask):
        background_color = get_most_frequent_color(image_array, background_mask)
        result_array[background_mask] = background_color
        cluster_colors[0] = background_color

    return Image.fromarray(result_array), cluster_colors


def process_fill_area_with_mask(region_mask_np, fill_image,
                                use_color_distance=True,
                                distance_threshold=10,
                                max_clusters=50):
    """
    region_maskを使用した塗り領域均一化処理

    Args:
        region_mask_np: ラベル用の画像（numpy array, HxW または HxWxC）
        fill_image: 塗り画像（PIL Image）
        use_color_distance: CIEDE2000色距離でクラスタリングするか
        distance_threshold: CIEDE2000距離の閾値（use_color_distance=Trueの時のみ使用）
        max_clusters: 最大クラスタ数（use_color_distance=Falseの時のみ使用）

    Returns:
        処理済みの画像, 領域数, クラスタ色情報, ラベル配列
    """
    # region_maskからラベル配列を取得
    labeled_array, num_features, _ = convert_mask_to_labels(region_mask_np)

    print(f"[FillAreaSimple] Initial region count: {num_features} regions")

    if num_features == 0:
        # 領域が見つからない場合は元の画像をそのまま返す
        return fill_image, 0, {}, labeled_array

    # 画像サイズの検証と調整
    mask_shape = labeled_array.shape[:2]
    if fill_image.size != (mask_shape[1], mask_shape[0]):
        fill_image = fill_image.resize((mask_shape[1], mask_shape[0]), Image.Resampling.LANCZOS)

    # クラスタリング方法の選択
    if use_color_distance:
        # CIEDE2000色距離ベースのクラスタリング
        print(f"[FillAreaSimple] Using CIEDE2000 color distance clustering (threshold={distance_threshold})")
        labeled_array, num_features = merge_similar_regions_by_color_distance(
            fill_image, labeled_array, num_features, distance_threshold
        )
        print(f"[FillAreaSimple] Merged to {num_features} regions")
    else:
        # K-meansクラスタリング（従来の方法）
        if num_features > max_clusters:
            print(f"[FillAreaSimple] Too many regions ({num_features}). Merging to {max_clusters} clusters using K-means...")
            labeled_array, num_features = merge_similar_regions(
                fill_image, labeled_array, num_features, max_clusters
            )
            print(f"[FillAreaSimple] Merged to {num_features} regions")

    # 各領域を最頻出色で塗りつぶし
    result_image, cluster_colors = fill_areas_simple(fill_image, labeled_array, num_features)

    return result_image, num_features, cluster_colors, labeled_array


class FillAreaSimpleNode:
    """
    シンプルな塗り領域均一化ノード
    カラーラベル画像やSplitAreaNodeからのregion_maskを使用して、各領域を単色で塗りつぶす
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill_image": ("IMAGE",),
                # ここを IMAGE に変更：RGBラベル画像もそのまま渡せる
                "region_mask": ("IMAGE",),
                "use_color_distance": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Use CIEDE2000 Color Distance"
                }),
                "distance_threshold": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "slider",
                    "display_label": "Color Distance Threshold"
                }),
                "max_clusters": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "display": "slider",
                    "display_label": "Max Clusters (K-means)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "CLUSTER_INFO")
    RETURN_NAMES = ("filled_image", "preview", "region_count", "cluster_info")

    FUNCTION = "execute"

    CATEGORY = "LayerDivider"

    def execute(self, fill_image, region_mask,
                use_color_distance=True, distance_threshold=10,
                max_clusters=30, **kwargs):
        """
        シンプルな塗り領域均一化処理を実行

        Args:
            fill_image: 塗り画像のテンソル
            region_mask: 領域マスク用画像（IMAGE）
            use_color_distance: CIEDE2000色距離でクラスタリングするか
            distance_threshold: CIEDE2000距離の閾値
            max_clusters: 最大クラスタ数（K-means用）
            **kwargs: 互換性のための追加引数（無視される）

        Returns:
            filled_image: 均一化された塗り画像
            preview: 処理前後の比較画像
            region_count: 検出された領域数
        """

        # 互換性のための警告メッセージ
        if kwargs:
            print(f"[FillAreaSimple] Warning: Ignoring unexpected arguments: {list(kwargs.keys())}")

        # 塗り画像をPIL Imageに変換
        fill_pil = tensor_to_pil(fill_image)

        # region_mask テンソルをPIL→numpyに変換
        mask_pil = tensor_to_pil(region_mask)
        mask_np = np.array(mask_pil)

        # region_maskを使用して処理
        result_image, num_features, cluster_colors, labeled_array = process_fill_area_with_mask(
            mask_np,
            fill_pil.convert("RGB"),
            use_color_distance,
            distance_threshold,
            max_clusters
        )

        # 比較用のプレビュー画像を作成
        preview = create_comparison_preview(fill_pil, result_image)

        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(result_image)
        preview_tensor = pil_to_tensor(preview)

        # クラスタ情報を辞書として返す
        cluster_info = {
            'colors': cluster_colors,
            'labeled_array': labeled_array,
            'num_regions': num_features
        }

        return (output_tensor, preview_tensor, num_features, cluster_info)


class FillAreaSimpleVisualizeNode:
    """
    シンプルな塗り領域均一化ノード（可視化機能付き）
    カラーラベル画像やSplitAreaNodeからのregion_maskを使用して、各領域の境界や領域番号を表示
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill_image": ("IMAGE",),
                # ここも IMAGE に変更
                "region_mask": ("IMAGE",),
                "visualization_mode": (["filled", "regions", "boundaries", "comparison"],),
                "show_region_numbers": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Show Region Numbers"
                }),
            }
        }

    # region_mask_out も IMAGE にしておく
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("filled_image", "visualization", "region_count", "region_mask_out")

    FUNCTION = "execute"

    CATEGORY = "LayerDivider"

    def execute(self, fill_image, region_mask,
                visualization_mode="filled",
                show_region_numbers=False, **kwargs):
        """
        シンプルな塗り領域均一化処理を実行（可視化機能付き）

        Args:
            fill_image: 塗り画像のテンソル
            region_mask: 領域マスク用画像（IMAGE）
            visualization_mode: 可視化モード
            show_region_numbers: 領域番号を表示するか

        Returns:
            filled_image: 均一化された塗り画像
            visualization: 選択された可視化
            region_count: 検出された領域数
            region_mask_out: 領域マスク（入力をそのまま出力）
        """

        # 塗り画像をPIL Imageに変換
        fill_pil = tensor_to_pil(fill_image)
        fill_rgb = fill_pil.convert("RGB")

        # region_mask テンソルをPIL→numpyに変換
        mask_pil = tensor_to_pil(region_mask)
        mask_np = np.array(mask_pil)

        labeled_array, num_features, _ = convert_mask_to_labels(mask_np)

        # 入力されたマスクをそのまま出力用に使用
        region_mask_tensor = region_mask

        # 塗りつぶし処理
        result_image, cluster_colors = fill_areas_simple(fill_rgb, labeled_array, num_features)

        # 可視化の作成
        if visualization_mode == "regions":
            visualization = visualize_regions(labeled_array, num_features, show_region_numbers)
        elif visualization_mode == "boundaries":
            visualization = visualize_boundaries(labeled_array)
        elif visualization_mode == "comparison":
            visualization = create_comparison_preview(fill_pil, result_image)
        else:  # "filled"
            visualization = result_image

        # テンソルに変換
        output_tensor = pil_to_tensor(result_image)
        visualization_tensor = pil_to_tensor(visualization)

        return (output_tensor, visualization_tensor, num_features, region_mask_tensor)


def create_comparison_preview(original, processed):
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


def visualize_regions(labeled_array, num_features, show_numbers=False):
    """領域を異なる色で可視化"""
    # カラーマップの作成（各領域に異なる色を割り当て）
    np.random.seed(42)  # 再現性のため
    colors = np.random.randint(50, 255, size=(num_features + 1, 3))
    colors[0] = [0, 0, 0]  # 背景は黒

    # ラベル画像をカラー画像に変換
    result = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    for label_id in range(num_features + 1):
        mask = labeled_array == label_id
        result[mask] = colors[label_id]

    result_image = Image.fromarray(result)

    # 領域番号を表示
    if show_numbers and num_features > 0:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(result_image)

        # 各領域の重心を計算して番号を表示
        for label_id in range(1, min(num_features + 1, 100)):  # 最大100個まで表示
            mask = labeled_array == label_id
            if np.any(mask):
                # 重心を計算
                coords = np.argwhere(mask)
                center_y, center_x = coords.mean(axis=0).astype(int)

                # 番号を描画
                draw.text(
                    (center_x, center_y),
                    str(label_id),
                    fill=(255, 255, 255),
                    anchor="mm"
                )

    return result_image


def visualize_boundaries(labeled_array):
    """領域の境界線を可視化"""
    from scipy import ndimage

    # エッジ検出で境界を抽出
    edges = ndimage.sobel(labeled_array)
    edges = (edges > 0).astype(np.uint8) * 255

    # 境界線を赤色で表示
    result = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    result[:, :, 0] = edges  # Red channel

    # 背景を薄いグレーにして境界を見やすくする
    background = labeled_array == 0
    result[background] = [50, 50, 50]

    return Image.fromarray(result)


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillAreaSimple": FillAreaSimpleNode,
    "FillAreaSimpleVisualize": FillAreaSimpleVisualizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillAreaSimple": "Fill Area Simple",
    "FillAreaSimpleVisualize": "Fill Area Simple (Visualize)",
}
