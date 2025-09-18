"""
Fill Space Node for ComfyUI
線画の隙間を最近傍の色で埋めるノード
BFS（幅優先探索）を使用して白い部分を最も近い黒い部分の色で塗りつぶす
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from collections import deque
from tqdm import tqdm
import folder_paths
import os

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


def find_nearest_black_pixel_bfs(mask, white_coords, progress_callback=None):
    """
    BFSを用いて、各白ピクセルの最も近い黒ピクセルを見つける
    
    Args:
        mask: マスク画像（黒=0、白=255）
        white_coords: 白ピクセルの座標リスト
        progress_callback: 進捗コールバック関数
    
    Returns:
        最近傍黒ピクセルの座標マッピング
    """
    # 画像サイズ
    if len(mask.shape) == 3:
        h, w, _ = mask.shape
    else:
        h, w = mask.shape
    
    # 方向ベクトル（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 最近傍黒ピクセルの座標マッピング
    nearest_black_coords = {}
    
    total_coords = len(white_coords)
    for idx, (y, x) in enumerate(white_coords):
        if progress_callback and idx % 100 == 0:
            progress_callback(idx / total_coords)
        
        # BFS用のキューを初期化
        queue = deque([(y, x)])
        visited = set()
        visited.add((y, x))
        
        # BFS開始
        while queue:
            cy, cx = queue.popleft()
            
            # 黒ピクセルが見つかった場合、座標を記録して終了
            if mask[cy, cx] == 0:
                nearest_black_coords[(y, x)] = (cy, cx)
                break
            
            # 隣接するピクセルを探索
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    queue.append((ny, nx))
                    visited.add((ny, nx))
    
    return nearest_black_coords


def find_nearest_black_pixel_bfs_optimized(mask, white_coords):
    """
    最適化されたBFS実装（複数の白ピクセルを同時に処理）
    
    Args:
        mask: マスク画像
        white_coords: 白ピクセルの座標
    
    Returns:
        最近傍黒ピクセルの座標マッピング
    """
    if len(mask.shape) == 3:
        h, w, _ = mask.shape
    else:
        h, w = mask.shape
    
    # 8方向（斜めも含む）で探索する場合はこちらを使用
    directions_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # 4方向のみの場合
    directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 使用する方向を選択
    directions = directions_4
    
    nearest_black_coords = {}
    
    # 黒ピクセルから開始して外側に向かって探索
    # これにより全体的な計算量を削減
    black_pixels = np.argwhere(mask == 0)
    
    if len(black_pixels) == 0:
        # 黒ピクセルがない場合は元の座標を返す
        return {tuple(coord): tuple(coord) for coord in white_coords}
    
    # 距離マップを作成
    distance_map = np.full((h, w), np.inf)
    nearest_map = np.zeros((h, w, 2), dtype=np.int32)
    
    # 黒ピクセルの位置を距離0として初期化
    queue = deque()
    for y, x in black_pixels:
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
                new_dist = current_dist + 1
                if new_dist < distance_map[ny, nx]:
                    distance_map[ny, nx] = new_dist
                    nearest_map[ny, nx] = nearest_map[cy, cx]
                    queue.append((ny, nx))
    
    # 白ピクセルに対して最近傍を取得
    for y, x in white_coords:
        nearest_black_coords[(y, x)] = tuple(nearest_map[y, x])
    
    return nearest_black_coords


def calc_fill_space(mask_image, source_image, use_optimized=True):
    """
    マスク画像の白い部分を、最近傍の黒い部分に対応する元画像の色で塗りつぶす
    
    Args:
        mask_image: マスク画像（線画）
        source_image: 元画像（塗り画像）
        use_optimized: 最適化版のBFSを使用するか
    
    Returns:
        処理済みの画像
    """
    # マスク画像を配列に変換
    mask_array = np.array(mask_image)
    
    # 白ピクセル（255）の座標を取得
    white_coords = np.argwhere(mask_array == 255)
    
    if len(white_coords) == 0:
        # 白ピクセルがない場合は元画像をそのまま返す
        return source_image
    
    # BFSを使用して白ピクセルの最近傍黒ピクセルの座標を計算
    if use_optimized:
        nearest_black_mapping = find_nearest_black_pixel_bfs_optimized(mask_array, white_coords)
    else:
        nearest_black_mapping = find_nearest_black_pixel_bfs(mask_array, white_coords)
    
    # 元画像の配列を取得
    source_array = np.array(source_image)
    
    # 処理後の画像を元画像と同じ形状で初期化
    output_array = source_array.copy()
    
    # 白ピクセルを最近傍黒ピクセルに対応する元画像の値に置き換え
    for (y, x), (black_y, black_x) in nearest_black_mapping.items():
        output_array[y, x] = source_array[black_y, black_x]
    
    return Image.fromarray(output_array)


class FillSpaceNode:
    """
    線画の隙間を最近傍の色で埋めるノード
    白い部分を最も近い黒い部分の色で塗りつぶす
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "flat_image": ("IMAGE",),
                "invert_binary": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Invert Binary Image"
                }),
                "use_optimized": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Use Optimized Algorithm"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "preview")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, flat_image, invert_binary=True, use_optimized=True):
        """
        線画の隙間を埋める処理を実行
        
        Args:
            binary_image: バイナリ画像（線画）のテンソル
            flat_image: 塗り画像のテンソル
            invert_binary: バイナリ画像を反転するか
            use_optimized: 最適化アルゴリズムを使用するか
        
        Returns:
            filled_image: 隙間が埋められた画像
            preview: 処理前後の比較画像
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        flat_pil = tensor_to_pil(flat_image)
        
        # バイナリ画像をグレースケールに変換
        if binary_pil.mode != 'L':
            binary_pil = binary_pil.convert('L')
        
        # 必要に応じてバイナリ画像を反転
        if invert_binary:
            binary_pil = ImageOps.invert(binary_pil)
        
        # 隙間を埋める処理
        filled_image = calc_fill_space(binary_pil, flat_pil, use_optimized)
        
        # プレビュー画像の作成（前後比較）
        preview = create_before_after_preview(flat_pil, filled_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(filled_image)
        preview_tensor = pil_to_tensor(preview)
        
        return (output_tensor, preview_tensor)


class FillSpaceAdvancedNode:
    """
    高度な隙間埋めノード
    より詳細な制御と可視化機能付き
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "flat_image": ("IMAGE",),
                "invert_binary": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Invert Binary Image"
                }),
                "search_directions": (["4-way", "8-way"],),
                "fill_mode": (["nearest", "average", "gradient"],),
                "threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Binary Threshold"
                }),
                "blur_result": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Blur Result"
                }),
                "blur_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 21,
                    "step": 2,
                    "display": "slider",
                    "display_label": "Blur Kernel Size"
                }),
                "output_mode": (["filled", "distance_map", "comparison", "overlay"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("filled_image", "visualization", "original", "fill_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, flat_image, invert_binary=True,
                search_directions="4-way", fill_mode="nearest", threshold=128,
                blur_result=False, blur_size=3, output_mode="filled"):
        """
        高度な隙間埋め処理
        
        Args:
            binary_image: バイナリ画像（線画）
            flat_image: 塗り画像
            invert_binary: バイナリ画像を反転
            search_directions: 探索方向（4方向または8方向）
            fill_mode: 塗りつぶしモード
            threshold: 二値化閾値
            blur_result: 結果をぼかすか
            blur_size: ぼかしのカーネルサイズ
            output_mode: 出力モード
        
        Returns:
            filled_image: 隙間が埋められた画像
            visualization: 可視化画像
            original: 元の塗り画像
            fill_mask: 塗りつぶしマスク
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        flat_pil = tensor_to_pil(flat_image)
        original = flat_pil.copy()
        
        # バイナリ画像をグレースケールに変換
        if binary_pil.mode != 'L':
            binary_pil = binary_pil.convert('L')
        
        # 閾値処理
        binary_array = np.array(binary_pil)
        binary_array = np.where(binary_array > threshold, 255, 0).astype(np.uint8)
        binary_pil = Image.fromarray(binary_array)
        
        # 必要に応じてバイナリ画像を反転
        if invert_binary:
            binary_pil = ImageOps.invert(binary_pil)
        
        # 探索方向の設定
        use_8way = (search_directions == "8-way")
        
        # 隙間を埋める処理
        if fill_mode == "nearest":
            filled_image = calc_fill_space_advanced(
                binary_pil, flat_pil, use_8way=use_8way
            )
        elif fill_mode == "average":
            filled_image = calc_fill_space_average(
                binary_pil, flat_pil, use_8way=use_8way
            )
        else:  # gradient
            filled_image = calc_fill_space_gradient(
                binary_pil, flat_pil, use_8way=use_8way
            )
        
        # ぼかし処理
        if blur_result:
            import cv2
            filled_array = np.array(filled_image)
            filled_array = cv2.GaussianBlur(filled_array, (blur_size, blur_size), 0)
            filled_image = Image.fromarray(filled_array)
        
        # 塗りつぶしマスクの作成
        fill_mask = create_fill_mask(binary_pil)
        
        # 可視化の作成
        if output_mode == "distance_map":
            visualization = create_distance_map_visualization(binary_pil)
        elif output_mode == "comparison":
            visualization = create_before_after_preview(original, filled_image)
        elif output_mode == "overlay":
            visualization = create_overlay_visualization(original, filled_image, fill_mask)
        else:  # "filled"
            visualization = filled_image
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(filled_image)
        visualization_tensor = pil_to_tensor(visualization)
        original_tensor = pil_to_tensor(original)
        
        # マスクをテンソルに変換
        mask_array = np.array(fill_mask, dtype=np.float32) / 255.0
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_tensor = torch.from_numpy(mask_array)
        
        return (output_tensor, visualization_tensor, original_tensor, mask_tensor)


def calc_fill_space_advanced(mask_image, source_image, use_8way=False):
    """
    高度な隙間埋め処理（8方向探索対応）
    """
    mask_array = np.array(mask_image)
    
    if len(mask_array.shape) == 3:
        h, w, _ = mask_array.shape
    else:
        h, w = mask_array.shape
    
    # 探索方向
    if use_8way:
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 白ピクセルの座標を取得
    white_coords = np.argwhere(mask_array == 255)
    
    if len(white_coords) == 0:
        return source_image
    
    # 距離マップの作成
    distance_map = np.full((h, w), np.inf)
    nearest_map = np.zeros((h, w, 2), dtype=np.int32)
    
    # 黒ピクセルから開始
    black_pixels = np.argwhere(mask_array == 0)
    queue = deque()
    
    for y, x in black_pixels:
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
                if dy != 0 and dx != 0:
                    new_dist = current_dist + 1.414
                else:
                    new_dist = current_dist + 1
                
                if new_dist < distance_map[ny, nx]:
                    distance_map[ny, nx] = new_dist
                    nearest_map[ny, nx] = nearest_map[cy, cx]
                    queue.append((ny, nx))
    
    # 元画像の配列を取得
    source_array = np.array(source_image)
    output_array = source_array.copy()
    
    # 白ピクセルを塗りつぶし
    for y, x in white_coords:
        nearest_y, nearest_x = nearest_map[y, x]
        output_array[y, x] = source_array[nearest_y, nearest_x]
    
    return Image.fromarray(output_array)


def calc_fill_space_average(mask_image, source_image, use_8way=False):
    """
    平均値による隙間埋め処理
    """
    # 基本的な処理は nearest と同じ
    filled = calc_fill_space_advanced(mask_image, source_image, use_8way)
    
    # 追加で周囲のピクセルとブレンド
    import cv2
    filled_array = np.array(filled)
    mask_array = np.array(mask_image)
    
    # 白い部分のみに平均フィルタを適用
    white_mask = (mask_array == 255)
    if np.any(white_mask):
        blurred = cv2.blur(filled_array, (3, 3))
        filled_array[white_mask] = blurred[white_mask]
    
    return Image.fromarray(filled_array)


def calc_fill_space_gradient(mask_image, source_image, use_8way=False):
    """
    グラデーション補間による隙間埋め処理
    """
    mask_array = np.array(mask_image)
    source_array = np.array(source_image)
    
    # インペインティング（OpenCVの修復機能）を使用
    import cv2
    
    # マスクを作成（白い部分が修復対象）
    inpaint_mask = (mask_array == 255).astype(np.uint8) * 255
    
    # インペインティング実行
    if len(source_array.shape) == 3:
        result = cv2.inpaint(source_array, inpaint_mask, 3, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(source_array, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    return Image.fromarray(result)


def create_fill_mask(binary_image):
    """塗りつぶしマスクの作成"""
    return binary_image


def create_distance_map_visualization(binary_image):
    """距離マップの可視化"""
    binary_array = np.array(binary_image)
    
    if len(binary_array.shape) == 3:
        h, w, _ = binary_array.shape
    else:
        h, w = binary_array.shape
    
    # 距離マップの作成
    distance_map = np.full((h, w), np.inf)
    black_pixels = np.argwhere(binary_array == 0)
    
    queue = deque()
    for y, x in black_pixels:
        distance_map[y, x] = 0
        queue.append((y, x))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        cy, cx = queue.popleft()
        current_dist = distance_map[cy, cx]
        
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                new_dist = current_dist + 1
                if new_dist < distance_map[ny, nx]:
                    distance_map[ny, nx] = new_dist
                    queue.append((ny, nx))
    
    # 正規化して可視化
    max_dist = np.max(distance_map[distance_map != np.inf])
    if max_dist > 0:
        distance_map[distance_map != np.inf] = (distance_map[distance_map != np.inf] / max_dist * 255)
    distance_map[distance_map == np.inf] = 255
    
    # カラーマップ適用
    distance_map_uint8 = distance_map.astype(np.uint8)
    import cv2
    colored = cv2.applyColorMap(distance_map_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(colored)


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
    
    return comparison


def create_overlay_visualization(original, filled, mask):
    """オーバーレイ可視化"""
    original_rgba = original.convert("RGBA")
    filled_rgba = filled.convert("RGBA")
    mask_array = np.array(mask)
    
    # マスクの部分だけ半透明で表示
    overlay = original_rgba.copy()
    filled_rgba.putalpha(128)
    
    # マスクの白い部分のみ合成
    if len(mask_array.shape) == 2:
        white_pixels = np.argwhere(mask_array == 255)
        for y, x in white_pixels:
            if y < overlay.height and x < overlay.width:
                filled_pixel = filled_rgba.getpixel((x, y))
                overlay.putpixel((x, y), filled_pixel)
    
    return overlay


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillSpace": FillSpaceNode,
    "FillSpace Advanced": FillSpaceAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillSpace": "Fill Space",
    "FillSpace Advanced": "Fill Space (Advanced)",
}
