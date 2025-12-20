"""
Simple PSD Layer Stack Node
複数の画像を受け取り、順番に重ねてPSDファイルとして出力するシンプルなノード
"""

import torch
import numpy as np
import os
from datetime import datetime
import folder_paths
from pytoshop.user import nested_layers
from pytoshop import enums


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
        height = first_image.shape[1]
        width = first_image.shape[2]
        num_layers = len(images_list)
        
        print(f"Creating PSD with {num_layers} layers, size: {width}x{height}")
        print(f"Layer order: image1 (bottom) → image{num_layers} (top)")
        
        # レイヤーリストを作成
        layers_list = []
        
        # 各画像をレイヤーとして追加（下から上へ）
        # image1 = 一番下のレイヤー（背景）
        # image10 = 一番上のレイヤー（前景）
        for i, image_tensor in enumerate(images_list):
            # バッチの最初の画像を取得（各入力は [B, H, W, C] 形式）
            img = image_tensor[0] if image_tensor.shape[0] > 0 else image_tensor
            
            # テンソルをNumPy配列に変換（0-255のuint8）
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            
            # チャンネル数を確認
            has_alpha = img_np.shape[2] == 4
            
            # RGBチャンネルを分離
            if has_alpha:
                channels = [
                    img_np[:, :, 0],  # R
                    img_np[:, :, 1],  # G
                    img_np[:, :, 2],  # B
                    img_np[:, :, 3]   # A
                ]
            else:
                channels = [
                    img_np[:, :, 0],  # R
                    img_np[:, :, 1],  # G
                    img_np[:, :, 2]   # B
                ]
            
            # レイヤー名を生成（入力番号を明示）
            layer_name = f"image{i + 1}"
            
            # レイヤーを作成
            layer = nested_layers.Image(
                name=layer_name,
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
        
        # PSDファイルを保存
        with open(filename, 'wb') as f:
            nested_layers.export(
                layers_list,
                f,
                width=width,
                height=height,
                depth=8,
                color_mode=enums.ColorMode.rgb
            )
        
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
