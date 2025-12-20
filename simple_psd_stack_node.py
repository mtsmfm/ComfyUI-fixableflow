"""
Simple PSD Layer Stack Node
3つの画像（base, shade, lineart）を受け取り、順番に重ねてPSDファイルとして出力し、
合成画像も返すノード
"""

import torch
import numpy as np
import os
from datetime import datetime
import folder_paths
from pytoshop import layers
import pytoshop
from pytoshop.enums import BlendMode


class SimplePSDStackNode:
    """
    3つの画像を受け取り、順番にレイヤーとして重ねてPSD出力 + 合成画像を返すノード
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
    FUNCTION = "create_psd"
    CATEGORY = "image/psd"
    OUTPUT_NODE = True
    
    def create_psd(self, base, shade, lineart, filename_prefix="layered"):
        """
        3つの画像をレイヤーとして重ねたPSDファイルを作成し、合成画像を返す
        
        Args:
            base: ベース画像（一番下）
            shade: 影画像（真ん中）
            lineart: 線画（一番上）
            filename_prefix: ファイル名のプレフィックス
        
        Returns:
            合成画像
        """
        # 画像リストを作成（下から上への順序）
        images_list = [base, shade, lineart]
        layer_names = ["base", "shade", "lineart"]
        
        # 出力ディレクトリの設定
        comfy_path = os.path.dirname(folder_paths.__file__)
        output_dir = os.path.join(comfy_path, 'output')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.psd")
        
        # 最初の画像から情報を取得
        first_image = images_list[0]
        img_sample = first_image[0] if first_image.shape[0] > 0 else first_image
        height = img_sample.shape[0]
        width = img_sample.shape[1]
        
        print(f"Creating PSD with 3 layers, size: {width}x{height}")
        print(f"Layer order: base (bottom) → shade (middle) → lineart (top)")
        
        # PSDファイルオブジェクトを作成
        psd = pytoshop.core.PsdFile(num_channels=3, height=height, width=width)
        
        # 合成用の画像を準備（NumPy配列）
        composite_layers = []
        
        # 各画像をレイヤーとして追加
        for i, (image_tensor, layer_name) in enumerate(zip(images_list, layer_names)):
            # バッチの最初の画像を取得
            img = image_tensor[0] if image_tensor.shape[0] > 0 else image_tensor
            
            # テンソルをNumPy配列に変換（0-1の範囲を0-255に変換）
            img_np = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # 合成用に保存
            composite_layers.append(img_np)
            
            # チャンネル数を確認
            has_alpha = img_np.shape[2] == 4
            
            # 各チャンネルのデータを作成
            if has_alpha:
                # RGBA画像の場合
                layer_alpha = layers.ChannelImageData(image=img_np[:, :, 3], compression=1)
                layer_r = layers.ChannelImageData(image=img_np[:, :, 0], compression=1)
                layer_g = layers.ChannelImageData(image=img_np[:, :, 1], compression=1)
                layer_b = layers.ChannelImageData(image=img_np[:, :, 2], compression=1)
                
                # レイヤーレコードを作成
                new_layer = layers.LayerRecord(
                    channels={-1: layer_alpha, 0: layer_r, 1: layer_g, 2: layer_b},
                    top=0,
                    bottom=height,
                    left=0,
                    right=width,
                    blend_mode=BlendMode.normal,
                    name=layer_name,
                    opacity=255
                )
            else:
                # RGB画像の場合（アルファなし）
                layer_r = layers.ChannelImageData(image=img_np[:, :, 0], compression=1)
                layer_g = layers.ChannelImageData(image=img_np[:, :, 1], compression=1)
                layer_b = layers.ChannelImageData(image=img_np[:, :, 2], compression=1)
                
                # レイヤーレコードを作成
                new_layer = layers.LayerRecord(
                    channels={0: layer_r, 1: layer_g, 2: layer_b},
                    top=0,
                    bottom=height,
                    left=0,
                    right=width,
                    blend_mode=BlendMode.normal,
                    name=layer_name,
                    opacity=255
                )
            
            # PSDにレイヤーを追加
            psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
        
        # PSDファイルを保存
        with open(filename, 'wb') as fd:
            psd.write(fd)
        
        # ログファイルにパスを保存（ダウンロードボタン用）
        log_path = os.path.join(output_dir, 'simple_psd_stack_savepath.log')
        with open(log_path, 'w') as log_file:
            log_file.write(os.path.basename(filename))
        
        print(f"PSD file saved: {filename}")
        print(f"Log file updated: {log_path}")
        
        # 3つの画像を合成（アルファブレンディング）
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
