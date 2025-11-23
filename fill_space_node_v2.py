"""
Fill Space Node V2 for ComfyUI (fixed)
線画の下のピクセルを、クラスタ化された色から最も近い色で塗りつぶすノード
- CIEDE2000色差計算を使用（オプション）
- 線画のアンチエイリアスを拾わないよう white_mask を見直し
- 色サンプルは original_image ではなく flat_image（線なしベタ塗り）から取得
"""

import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
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
    rgb_normalized = np.array(rgb_color).reshape(1, 1, 3) / 255.0
    lab = skcolor.rgb2lab(rgb_normalized)
    return lab[0, 0]


def ciede2000_distance(lab1, lab2):
    """CIEDE2000色差を計算"""
    lab1_reshaped = np.array(lab1).reshape(1, 1, 3)
    lab2_reshaped = np.array(lab2).reshape(1, 1, 3)
    delta_e = skcolor.deltaE_ciede2000(lab1_reshaped, lab2_reshaped)
    return float(delta_e[0, 0])


def find_closest_cluster_color(pixel_color, cluster_colors):
    """
    ピクセルの色に最も近いクラスタ色を見つける（未使用だが残しておく）
    """
    if not cluster_colors:
        return pixel_color

    pixel_lab = rgb_to_lab(pixel_color)

    min_distance = float('inf')
    closest_color = pixel_color

    for cluster_id, cluster_color in cluster_colors.items():
        cluster_lab = rgb_to_lab(cluster_color)
        distance = ciede2000_distance(pixel_lab, cluster_lab)

        if distance < min_distance:
            min_distance = distance
            closest_color = cluster_color

    return closest_color


def process_fill_space_with_clusters_progress(
    binary_image,
    original_image,
    cluster_info,
    invert_binary=True,
    progress_callback=None,
    flat_image=None,
):
    """
    線画の下のピクセルをクラスタ色で塗りつぶす（K-means最適化版・逐次版）

    重要な変更点:
      - 塗りの元データは original_image ではなく flat_image を使用する
      - white_mask は >= 250 のしきい値で決定し、線のアンチエイリアスを除外
    """
    from sklearn.cluster import KMeans

    # バイナリ画像をグレースケールに変換
    if binary_image.mode != 'L':
        binary_gray = binary_image.convert('L')
    else:
        binary_gray = binary_image

    # 必要に応じてバイナリ画像を反転
    if invert_binary:
        binary_gray = ImageOps.invert(binary_gray)

    binary_array = np.array(binary_gray)

    # ベース画像としてflat_imageを使用（なければoriginal_imageを使用）
    if flat_image is not None:
        base_array = np.array(flat_image.convert('RGB'))  # 線なしベタ塗り
    else:
        base_array = np.array(original_image.convert('RGB'))

    # クラスタ色情報を取得
    cluster_colors = cluster_info.get('colors', {})
    num_clusters = len(cluster_colors)

    if num_clusters == 0:
        if flat_image is not None:
            return flat_image
        return original_image

    # 出力画像を初期化（flat_imageのコピー）
    output_array = base_array.copy()

    # 白ピクセル（＝塗りつぶし対象）のマスク
    # 255 一致ではなく 250 以上を採用して、アンチエイリアスされた線のグレーを除外
    white_mask = binary_array >= 250
    total_pixels = int(np.sum(white_mask))

    print(f"[FillSpaceV2] Processing {total_pixels} pixels under line art")
    print(f"[FillSpaceV2] Using {num_clusters} color clusters from Fill Area Simple")

    if total_pixels == 0:
        return Image.fromarray(output_array.astype(np.uint8))

    # 線画下のピクセル色は original ではなく base_array から取得（線なしベタ）
    white_pixel_colors = base_array[white_mask]

    # 線画下のピクセルを同じクラスタ数でK-meansクラスタリング
    print(f"[FillSpaceV2] Clustering fill pixels into {num_clusters} clusters...")
    unique_colors = np.unique(white_pixel_colors, axis=0)
    n_clusters = min(num_clusters, len(unique_colors))
    if n_clusters <= 0:
        return Image.fromarray(output_array.astype(np.uint8))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=1,  # 高速化
    )
    line_art_labels = kmeans.fit_predict(white_pixel_colors)
    line_art_centers = kmeans.cluster_centers_

    print(f"[FillSpaceV2] Created {len(line_art_centers)} line-art-side clusters")

    # Fill Areaのクラスタ色を配列に変換
    fill_cluster_ids = list(cluster_colors.keys())
    fill_cluster_array = np.array([cluster_colors[cid] for cid in fill_cluster_ids])

    # クラスタ間のマッピングを計算（クラスタ中心同士の比較：RGB距離）
    print(
        f"[FillSpaceV2] Computing cluster mappings "
        f"({len(line_art_centers)} x {len(fill_cluster_array)} comparisons)..."
    )
    cluster_mapping = np.zeros(len(line_art_centers), dtype=np.int32)

    for i, center in enumerate(line_art_centers):
        distances = np.sum((fill_cluster_array - center) ** 2, axis=1)
        cluster_mapping[i] = int(np.argmin(distances))

    print(f"[FillSpaceV2] Applying mapped colors to pixels...")

    # 各ピクセルに対してマッピングされた色を適用
    for pixel_idx in range(len(white_pixel_colors)):
        line_cluster_id = line_art_labels[pixel_idx]
        fill_cluster_idx = cluster_mapping[line_cluster_id]
        mapped_color = fill_cluster_array[fill_cluster_idx]
        white_pixel_colors[pixel_idx] = mapped_color

        if progress_callback is not None and pixel_idx % 10000 == 0:
            progress_callback(pixel_idx, len(white_pixel_colors))

    # 結果を出力画像に反映
    output_array[white_mask] = white_pixel_colors

    print(f"[FillSpaceV2] Completed processing")

    return Image.fromarray(output_array.astype(np.uint8))


def find_closest_cluster_color_optimized(pixel_color, cluster_colors, cluster_labs):
    """
    最適化版：事前計算されたLAB値を使用してピクセルの色に最も近いクラスタ色を見つける
    （今回の修正版では未使用だが、将来用に残しておく）
    """
    if not cluster_colors:
        return pixel_color

    pixel_lab = rgb_to_lab(pixel_color)

    min_distance = float('inf')
    closest_color = pixel_color

    for cluster_id, cluster_lab in cluster_labs.items():
        distance = ciede2000_distance(pixel_lab, cluster_lab)

        if distance < min_distance:
            min_distance = distance
            closest_color = cluster_colors[cluster_id]

    return closest_color


def process_fill_space_batch_optimized(
    binary_image,
    original_image,
    cluster_info,
    invert_binary=True,
    flat_image=None,
):
    """
    バッチ処理最適化版：K-meansでクラスタ間マッピング（超高速版）

    重要な変更点:
      - 塗り情報のソースは original_image ではなく flat_image
      - white_mask は >= 250 で決定して線のアンチエイリアスを除外
    """
    from sklearn.cluster import KMeans

    # バイナリ画像をグレースケールに変換
    if binary_image.mode != 'L':
        binary_gray = binary_image.convert('L')
    else:
        binary_gray = binary_image

    if invert_binary:
        binary_gray = ImageOps.invert(binary_gray)

    binary_array = np.array(binary_gray)

    # ベース画像としてflat_imageを使用（なければoriginal_imageを使用）
    if flat_image is not None:
        base_array = np.array(flat_image.convert('RGB'))
    else:
        base_array = np.array(original_image.convert('RGB'))

    # クラスタ色情報
    cluster_colors = cluster_info.get('colors', {})
    num_clusters = len(cluster_colors)

    if num_clusters == 0:
        if flat_image is not None:
            return flat_image
        return original_image

    output_array = base_array.copy()

    # 塗り対象のマスク
    white_mask = binary_array >= 250
    total_pixels = int(np.sum(white_mask))

    print(f"[FillSpaceV2] Batch processing {total_pixels} pixels")
    print(f"[FillSpaceV2] Using {num_clusters} color clusters")

    if total_pixels == 0:
        return Image.fromarray(output_array.astype(np.uint8))

    # 塗り対象ピクセルの色（線なしベタ）を取得
    white_pixel_colors = base_array[white_mask]

    # K-meansでクラスタリング
    print(f"[FillSpaceV2] K-means clustering into {num_clusters} clusters...")
    unique_colors = np.unique(white_pixel_colors, axis=0)
    n_clusters = min(num_clusters, len(unique_colors))
    if n_clusters <= 0:
        return Image.fromarray(output_array.astype(np.uint8))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    line_art_labels = kmeans.fit_predict(white_pixel_colors)
    line_art_centers = kmeans.cluster_centers_

    # Fill Areaクラスタ色を配列に変換
    cluster_ids = list(cluster_colors.keys())
    cluster_rgb_array = np.array([cluster_colors[cid] for cid in cluster_ids])

    # クラスタ中心間のマッピング（ベクトル化）
    print(f"[FillSpaceV2] Computing optimal cluster mapping...")
    distances = np.sum(
        (line_art_centers[:, np.newaxis, :] - cluster_rgb_array[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    cluster_mapping = np.argmin(distances, axis=1)

    # マッピングされた色を一括適用
    mapped_colors = cluster_rgb_array[cluster_mapping]
    result_colors = mapped_colors[line_art_labels]

    output_array[white_mask] = result_colors

    print(f"[FillSpaceV2] Batch processing completed")

    return Image.fromarray(output_array.astype(np.uint8))


class FillSpaceV2Node:
    """
    線画の下のピクセルをクラスタ色で塗りつぶすノード（修正版）

    - binary_image: 線画 or マスク
    - flat_image: バケツ塗りされたベース画像（線なしベタ）
    - original_image: 参考用の元画像（現在は色の取得には使用しない）
    - cluster_info: Fill Area Simple からのクラスタ情報
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),   # 線画（1枚目）
                "flat_image": ("IMAGE",),     # バケツ塗りされた画像（2枚目）
                "original_image": ("IMAGE",), # 元の塗り画像（3枚目）
                "cluster_info": ("CLUSTER_INFO",),
                "invert_binary": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Invert Binary Image"
                }),
                "use_batch_processing": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Use Batch Processing (Faster)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "preview")

    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(
        self,
        binary_image,
        flat_image,
        original_image,
        cluster_info,
        invert_binary=True,
        use_batch_processing=False,
    ):
        """
        線画の下のピクセルをクラスタ色で塗りつぶす処理
        """

        # テンソル → PIL
        binary_pil = tensor_to_pil(binary_image)
        flat_pil = tensor_to_pil(flat_image)
        original_pil = tensor_to_pil(original_image)

        if use_batch_processing:
            print("[FillSpaceV2] Using batch processing mode")
            filled_image = process_fill_space_batch_optimized(
                binary_pil,
                original_pil,
                cluster_info,
                invert_binary,
                flat_pil,  # ベースは常に flat_image
            )
        else:
            print("[FillSpaceV2] Using standard processing mode")
            filled_image = process_fill_space_with_clusters_progress(
                binary_pil,
                original_pil,
                cluster_info,
                invert_binary,
                None,
                flat_pil,  # ベースは常に flat_image
            )

        # プレビュー画像（前後比較）
        preview = create_before_after_preview(flat_pil, filled_image)

        # PIL → テンソル
        output_tensor = pil_to_tensor(filled_image)
        preview_tensor = pil_to_tensor(preview)

        return (output_tensor, preview_tensor)


def create_before_after_preview(original, processed):
    """処理前後の比較画像を作成"""
    width = original.width
    height = original.height

    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original.convert('RGB'), (0, 0))
    comparison.paste(processed.convert('RGB'), (width, 0))

    draw = ImageDraw.Draw(comparison)
    draw.line([(width, 0), (width, height)], fill=(255, 0, 0), width=2)

    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((width + 10, 10), "Processed", fill=(255, 255, 255))

    return comparison


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillSpaceV2": FillSpaceV2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillSpaceV2": "Fill Space V2 (Cluster-based, Fixed)",
}
