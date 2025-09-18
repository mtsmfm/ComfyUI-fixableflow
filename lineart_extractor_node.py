"""
ComfyUI Node for extracting line art with background removal
"""
from PIL import Image, ImageFilter
import numpy as np
import torch
import cv2
import os
import folder_paths

comfy_path = os.path.dirname(folder_paths.__file__)
lineart_extractor_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'
output_dir = f"{lineart_extractor_path}/output"

if not os.path.exists(f'{output_dir}'):
    os.makedirs(f'{output_dir}')


def convert_non_white_to_black(image):
    """
    Convert pixels that are not completely white to black
    """
    # Convert image to NumPy array
    image_np = np.array(image)
    
    # Set pixels with value > 200 to white (255)
    image_np[image_np > 200] = 255
    
    # Convert NumPy array back to image
    return Image.fromarray(image_np)


def process_lineart(image):
    """
    Process line art image: convert to grayscale, apply threshold, smooth, and create alpha channel
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Apply threshold to convert non-white pixels to black
    image = convert_non_white_to_black(image)
    
    # Apply smooth filter to reduce noise
    image = image.filter(ImageFilter.SMOOTH)
    
    # Create RGBA image with alpha channel based on grayscale values
    result = Image.new('RGBA', image.size)
    
    for x in range(image.width):
        for y in range(image.height):
            # Get grayscale value
            gray = image.getpixel((x, y))
            
            # Set alpha transparency
            if gray == 0:
                alpha = 0  # Completely transparent (black areas)
            else:
                alpha = 255 - gray  # Opacity based on grayscale value
            
            # Set pixel in new image (black with varying alpha)
            result.putpixel((x, y), (0, 0, 0, alpha))
    
    return result


def pil_to_comfy_image(pil_image):
    """
    Convert PIL Image to ComfyUI image tensor format
    """
    # Convert RGBA to RGB + Alpha
    if pil_image.mode == 'RGBA':
        # Split RGBA channels
        r, g, b, a = pil_image.split()
        # Convert to RGB
        rgb_image = Image.merge('RGB', (r, g, b))
        # Convert to numpy array
        np_image = np.array(rgb_image).astype(np.float32) / 255.0
    else:
        np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension
    np_image = np.expand_dims(np_image, axis=0)
    
    # Convert to torch tensor
    return torch.from_numpy(np_image)


def pil_to_comfy_mask(pil_image):
    """
    Convert PIL Image alpha channel to ComfyUI mask format
    """
    if pil_image.mode == 'RGBA':
        # Extract alpha channel
        alpha = pil_image.split()[3]
        # Convert to numpy array
        np_mask = np.array(alpha).astype(np.float32) / 255.0
        # Add batch dimension
        np_mask = np.expand_dims(np_mask, axis=0)
        # Convert to torch tensor
        return torch.from_numpy(np_mask)
    else:
        # Create full opacity mask if no alpha channel
        np_mask = np.ones((1, pil_image.height, pil_image.width), dtype=np.float32)
        return torch.from_numpy(np_mask)


class ExtractLineArt:
    """
    ComfyUI node for extracting line art from an image with background removal
    Processes RGB line drawings and outputs transparent line art
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "white_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "apply_smoothing": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("line_art", "alpha_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, image, white_threshold=200, apply_smoothing=True):
        """
        Execute the line art extraction process
        
        Args:
            image: Input image tensor from ComfyUI
            white_threshold: Threshold value for determining white pixels (0-255)
            apply_smoothing: Whether to apply smoothing filter
        
        Returns:
            Tuple of (processed_image, alpha_mask) as tensors
        """
        # Convert ComfyUI image to numpy array
        img_batch_np = image.cpu().detach().numpy().__mul__(255.).astype(np.uint8)
        
        # Process first image in batch
        input_image = Image.fromarray(img_batch_np[0])
        
        # Convert to grayscale
        gray_image = input_image.convert('L')
        
        # Apply custom threshold
        gray_np = np.array(gray_image)
        gray_np[gray_np > white_threshold] = 255
        gray_image = Image.fromarray(gray_np)
        
        # Apply smoothing if requested
        if apply_smoothing:
            gray_image = gray_image.filter(ImageFilter.SMOOTH)
        
        # Create RGBA image with alpha channel based on grayscale values
        result = Image.new('RGBA', gray_image.size)
        
        # Use numpy for faster processing
        gray_array = np.array(gray_image)
        result_array = np.zeros((gray_image.height, gray_image.width, 4), dtype=np.uint8)
        
        # Set RGB to black (0, 0, 0)
        result_array[:, :, :3] = 0
        
        # Set alpha channel
        # Where gray is 0 (black), alpha should be 0 (transparent)
        # Where gray is 255 (white), alpha should be 0 (transparent background)
        # Otherwise, alpha should be 255 - gray (line opacity)
        alpha_channel = np.where(gray_array == 0, 0, 255 - gray_array)
        result_array[:, :, 3] = alpha_channel
        
        # Convert back to PIL Image
        result = Image.fromarray(result_array, 'RGBA')
        
        # Convert to ComfyUI format
        line_art = pil_to_comfy_image(result)
        alpha_mask = pil_to_comfy_mask(result)
        
        return (line_art, alpha_mask)


class ExtractLineArtAdvanced:
    """
    Advanced version of ExtractLineArt with additional processing options
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "white_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "black_threshold": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "smoothing_iterations": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "display": "slider"
                }),
                "edge_detection": ("BOOLEAN", {
                    "default": False
                }),
                "invert_result": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("line_art", "alpha_mask", "preview")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, image, white_threshold=200, black_threshold=50, 
                smoothing_iterations=1, edge_detection=False, invert_result=False):
        """
        Execute the advanced line art extraction process
        """
        # Convert ComfyUI image to numpy array
        img_batch_np = image.cpu().detach().numpy().__mul__(255.).astype(np.uint8)
        
        # Process first image in batch
        input_image = Image.fromarray(img_batch_np[0])
        
        # Convert to grayscale
        gray_image = input_image.convert('L')
        gray_np = np.array(gray_image)
        
        # Apply edge detection if requested
        if edge_detection:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_np, black_threshold, white_threshold)
            gray_np = 255 - edges  # Invert edges (white background, black lines)
        else:
            # Apply dual threshold
            # Set very dark pixels to black
            gray_np[gray_np < black_threshold] = 0
            # Set very light pixels to white
            gray_np[gray_np > white_threshold] = 255
        
        gray_image = Image.fromarray(gray_np)
        
        # Apply smoothing iterations
        for _ in range(smoothing_iterations):
            gray_image = gray_image.filter(ImageFilter.SMOOTH)
        
        # Create RGBA image with alpha channel
        result = Image.new('RGBA', gray_image.size)
        gray_array = np.array(gray_image)
        result_array = np.zeros((gray_image.height, gray_image.width, 4), dtype=np.uint8)
        
        # Set RGB to black (0, 0, 0)
        result_array[:, :, :3] = 0
        
        # Calculate alpha channel
        if invert_result:
            # Invert: white becomes opaque, black becomes transparent
            alpha_channel = gray_array
        else:
            # Normal: black becomes opaque, white becomes transparent
            alpha_channel = 255 - gray_array
        
        # Apply alpha where original was not pure white
        alpha_channel = np.where(gray_array == 255, 0, alpha_channel)
        result_array[:, :, 3] = alpha_channel
        
        # Convert back to PIL Image
        result = Image.fromarray(result_array, 'RGBA')
        
        # Create preview image with white background
        preview = Image.new('RGBA', result.size, (255, 255, 255, 255))
        preview.paste(result, (0, 0), result)
        preview = preview.convert('RGB')
        
        # Convert to ComfyUI format
        line_art = pil_to_comfy_image(result)
        alpha_mask = pil_to_comfy_mask(result)
        preview_image = pil_to_comfy_image(preview)
        
        return (line_art, alpha_mask, preview_image)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ExtractLineArt": ExtractLineArt,
    "ExtractLineArtAdvanced": ExtractLineArtAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractLineArt": "Extract Line Art",
    "ExtractLineArtAdvanced": "Extract Line Art (Advanced)",
}
