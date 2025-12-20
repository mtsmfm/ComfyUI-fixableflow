"""
Shadow Extract Node (HSV Version)
影抽出ノード - HSV空間で影を検出してアルファチャンネルとして出力

- input1: 影あり画像
- input2: 影なし画像（フラット）
- 明度低下（V）と彩度上昇（S）にのみ反応
- 色相差は無視
- 出力: RGBA画像（RGB=影あり画像, Alpha=影量）
"""

import torch
import numpy as np
import cv2


class ShadowExtractNode:
    """
    2枚の画像から影を抽出してアルファチャンネルとして出力するノード
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE",),
                "input2": ("IMAGE",),
                "weight_V": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "weight_S": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "normalize_factor": ("FLOAT", {
                    "default": 40.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 1.0,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "extract_shadow"
    CATEGORY = "image/processing"
    
    def extract_shadow(self, input1, input2, weight_V=1.0, weight_S=0.5, normalize_factor=40.0):
        """
        影を抽出してRGBA画像として出力
        """
        from PIL import Image
        
        img1 = input1[0].clone()
        img2 = input2[0].clone()
        
        if img1.shape[2] == 4:
            img1 = img1[:, :, :3]
        if img2.shape[2] == 4:
            img2 = img2[:, :, :3]
        
        if img1.shape[:2] != img2.shape[:2]:
            img2_np = (img2.cpu().numpy() * 255).astype(np.uint8)
            img2_pil = Image.fromarray(img2_np)
            target_size = (img1.shape[1], img1.shape[0])
            img2_pil = img2_pil.resize(target_size, Image.LANCZOS)
            img2 = torch.from_numpy(np.array(img2_pil).astype(np.float32) / 255.0)
        
        img1_np = (img1.cpu().numpy() * 255).astype(np.uint8)
        img2_np = (img2.cpu().numpy() * 255).astype(np.uint8)
        
        hsv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        _, S1, V1 = cv2.split(hsv1)
        _, S2, V2 = cv2.split(hsv2)
        
        delta_V = np.maximum(0.0, V2 - V1)
        delta_S = np.maximum(0.0, S1 - S2)
        
        shadow_score = weight_V * delta_V + weight_S * delta_S
        alpha = np.clip(shadow_score / normalize_factor, 0.0, 1.0)
        alpha = alpha[:, :, np.newaxis]
        
        img1_float = img1.cpu().numpy()
        rgba = np.concatenate([img1_float, alpha], axis=2)
        
        result = torch.from_numpy(rgba.astype(np.float32))
        result = result.unsqueeze(0)
        
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ShadowExtractNode": ShadowExtractNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShadowExtractNode": "Shadow Extract (HSV)"
}
