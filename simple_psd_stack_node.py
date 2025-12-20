"""
Simple PSD Layer Stack Node
複数の画像を受け取り、順番に重ねてPSDファイルとして出力するシンプルなノード
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
    複数の画像を受け取り、順番にレイヤーとして重ねてPSD出力するノード
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 1枚目（一番下のレイヤー）
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "layered",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("psd_path",)
    FUNCTION = "create_psd"
    CATEGORY = "image/psd"
    OUTPUT_NODE = True
    
    def create_psd(self, image1, filename_prefix="layered", image2=None, image3=None, 
                   image4=None, image5=None, image6=None, image7=None, 
                   image8=None, image9=None, image10=None):
        """
        複数の画像をレイヤーとして重ねたPSDファイルを作成
        
        Args:
            image1-10: 個別の画像入力（image1は必須、他はオプション）
            filename_prefix: ファイル名のプレフィックス
        
        Returns:
            PSDファイルのパス
        """
        # 提供された画像を収集
        images_list = [image1]
        for img in [image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            if img is not None:
                images_list.append(img)
        
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
        # バッチの最初の画像を取得
        img_sample = first_image[0] if first_image.shape[0] > 0 else first_image
        height = img_sample.shape[0]
        width = img_sample.shape[1]
        num_layers = len(images_list)
        
        print(f"Creating PSD with {num_layers} layers, size: {width}x{height}")
        print(f"Layer order: image1 (bottom) → image{num_layers} (top)")
        
        # PSDファイルオブジェクトを作成（既存のld_utilsと同じ方法）
        psd = pytoshop.core.PsdFile(num_channels=3, height=height, width=width)
        
        # 各画像をレイヤーとして追加（下から上へ）
        # image1 = 一番下のレイヤー（背景）
        # image10 = 一番上のレイヤー（前景）
        for i, image_tensor in enumerate(images_list):
            # バッチの最初の画像を取得（各入力は [B, H, W, C] 形式）
            img = image_tensor[0] if image_tensor.shape[0] > 0 else image_tensor
            
            # テンソルをNumPy配列に変換（0-1の範囲を0-255に変換）
            img_np = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # チャンネル数を確認
            has_alpha = img_np.shape[2] == 4
            
            # レイヤー名を生成
            layer_name = f"image{i + 1}"
            
            # 各チャンネルのデータを作成（既存のld_utilsと同じ方法）
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
            # ファイル名のみを保存（パスなし）
            log_file.write(os.path.basename(filename))
        
        print(f"PSD file saved: {filename}")
        print(f"Total layers: {num_layers}")
        print(f"Log file updated: {log_path}")
        
        return (filename,)


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    "SimplePSDStackNode": SimplePSDStackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplePSDStackNode": "Simple PSD Stack"
}
