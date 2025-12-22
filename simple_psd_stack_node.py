"""
Simple PSD Layer Stack Node (Frontend PSD Generation)
3つの画像（base, shade, lineart）を受け取り、前端でPSD生成するためのデータを準備するノード
合成画像も返す
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
import folder_paths
from PIL import Image


class SimplePSDStackNode:
    """
    3つの画像を受け取り、前端でPSD生成するためのデータを準備するノード
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),      # ベース画像（一番下）
                "shade": ("IMAGE",),     # 影画像（真ん中）
                "lineart": ("IMAGE",),   # 線画（一番上）
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "layered",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "prepare_layers"
    CATEGORY = "FixableFlow"
    OUTPUT_NODE = True

    def prepare_layers(self, base, shade, lineart, filename_prefix="layered"):
        """
        3つの画像をレイヤーとして準備し、前端でPSD生成するための情報を保存

        Args:
            base: ベース画像（一番下）
            shade: 影画像（真ん中）
            lineart: 線画（一番上）
            filename_prefix: ファイル名のプレフィックス

        Returns:
            合成画像
        """
        images_list = [base, shade, lineart]
        layer_names = ["base", "shade", "lineart"]

        output_dir = folder_paths.get_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 画像サイズを取得
        first_image = images_list[0]
        img_sample = first_image[0] if first_image.shape[0] > 0 else first_image
        height = img_sample.shape[0]
        width = img_sample.shape[1]

        print(f"Preparing layers for frontend PSD generation, size: {width}x{height}")
        print(f"Layer order: base (bottom) → shade (middle) → lineart (top)")

        # 各レイヤーを一時PNGとして保存
        layer_info = []
        composite_layers = []

        for i, (image_tensor, layer_name) in enumerate(zip(images_list, layer_names)):
            # バッチの最初の画像を取得
            img = image_tensor[0] if image_tensor.shape[0] > 0 else image_tensor

            # テンソルをNumPy配列に変換（0-1の範囲を0-255に変換）
            img_np = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            composite_layers.append(img_np)

            # PNGとして保存
            pil_img = Image.fromarray(img_np)
            filename = f"{filename_prefix}_{timestamp}_{layer_name}.png"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath)

            layer_info.append({
                "name": layer_name,
                "filename": filename
            })

            print(f"  Layer saved: {filename}")

        # レイヤー情報をJSONとして保存（前端が読み取る）
        info_filename = f"{filename_prefix}_{timestamp}_layers.json"
        info_file = os.path.join(output_dir, info_filename)
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump({
                "prefix": filename_prefix,
                "timestamp": timestamp,
                "layers": layer_info,
                "width": int(width),
                "height": int(height)
            }, f, indent=2)

        # 最新のinfo fileパスを保存（前端がこのファイルを読んで最新のJSONを見つける）
        log_path = os.path.join(output_dir, 'simple_psd_stack_info.log')
        with open(log_path, 'w') as f:
            f.write(info_filename)

        print(f"Layer info saved: {info_filename}")
        print(f"Frontend can now generate PSD from these layers")

        # 3つの画像を合成
        composite = self.composite_images(composite_layers[0], composite_layers[1], composite_layers[2])

        # ComfyUI形式のテンソルに変換
        composite_tensor = torch.from_numpy(composite.astype(np.float32) / 255.0).unsqueeze(0)

        return (composite_tensor,)

    def composite_images(self, base, shade, lineart):
        """
        3つの画像を合成する

        Args:
            base: ベース画像（NumPy配列、uint8）
            shade: 影画像（NumPy配列、uint8）
            lineart: 線画（NumPy配列、uint8）

        Returns:
            合成画像（NumPy配列、uint8、RGB）
        """
        # float32に変換（0-1の範囲）
        base_f = base.astype(np.float32) / 255.0
        shade_f = shade.astype(np.float32) / 255.0
        lineart_f = lineart.astype(np.float32) / 255.0

        # ベース画像をRGBに変換（アルファがあれば削除）
        if base_f.shape[2] == 4:
            base_rgb = base_f[:, :, :3]
        else:
            base_rgb = base_f

        # 影を合成（shadeにアルファがあればアルファブレンディング、なければそのまま重ねる）
        if shade_f.shape[2] == 4:
            shade_rgb = shade_f[:, :, :3]
            shade_alpha = shade_f[:, :, 3:4]
            result = base_rgb * (1 - shade_alpha) + shade_rgb * shade_alpha
        else:
            # アルファがない場合はそのまま合成
            result = shade_f

        # 線画を合成（lineartにアルファがあればアルファブレンディング）
        if lineart_f.shape[2] == 4:
            lineart_rgb = lineart_f[:, :, :3]
            lineart_alpha = lineart_f[:, :, 3:4]
            result = result * (1 - lineart_alpha) + lineart_rgb * lineart_alpha
        else:
            # アルファがない場合は乗算合成（線画っぽく）
            result = result * lineart_f

        # 0-255の範囲に戻してuint8に変換
        result = (result * 255.0).clip(0, 255).astype(np.uint8)

        return result


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    "SimplePSDStackNode": SimplePSDStackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplePSDStackNode": "Simple PSD Stack"
}
