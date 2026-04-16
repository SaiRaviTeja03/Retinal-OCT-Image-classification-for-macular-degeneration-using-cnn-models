import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import os
from typing import Tuple, Optional, Union

class OCTImageProcessor:
    """Utility class for processing OCT images"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image and convert to RGB
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image to target size"""
        if target_size is None:
            target_size = self.target_size
            
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            # Resize using LANCZOS for better quality
            resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            return np.array(resized_image)
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1] range"""
        try:
            return image.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error normalizing image: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Enhance image contrast"""
        try:
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(factor)
            return np.array(enhanced_image)
        except Exception as e:
            print(f"Error enhancing contrast: {e}")
            return image
    
    def enhance_brightness(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Enhance image brightness"""
        try:
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced_image = enhancer.enhance(factor)
            return np.array(enhanced_image)
        except Exception as e:
            print(f"Error enhancing brightness: {e}")
            return image
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian blur to reduce noise"""
        try:
            pil_image = Image.fromarray(image)
            blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=kernel_size))
            return np.array(blurred_image)
        except Exception as e:
            print(f"Error applying Gaussian blur: {e}")
            return image
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using median filter"""
        try:
            # Convert to grayscale for noise detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Apply median filter
            denoised_gray = cv2.medianBlur(gray, 3)
            # Convert back to RGB
            denoised_rgb = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)
            return denoised_rgb
        except Exception as e:
            print(f"Error removing noise: {e}")
            return image
    
    def crop_center(self, image: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """Crop center portion of image"""
        try:
            height, width = image.shape[:2]
            crop_height = int(height * crop_ratio)
            crop_width = int(width * crop_ratio)
            
            start_y = (height - crop_height) // 2
            start_x = (width - crop_width) // 2
            
            cropped_image = image[start_y:start_y + crop_height, 
                                start_x:start_x + crop_width]
            return cropped_image
        except Exception as e:
            print(f"Error cropping image: {e}")
            return image
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            
            # Merge channels back
            lab_clahe = cv2.merge([l_clahe, a, b])
            # Convert back to RGB
            rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            return rgb_clahe
        except Exception as e:
            print(f"Error applying CLAHE: {e}")
            return image
    
    def preprocess_for_model(self, image_path: str, enhance: bool = True) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline for model input"""
        try:
            # Load image
            image = self.load_image(image_path)
            if image is None:
                return None
            
            # Enhancement steps
            if enhance:
                # Apply CLAHE for better contrast
                image = self.apply_clahe(image)
                # Enhance contrast slightly
                image = self.enhance_contrast(image, factor=1.2)
                # Remove noise
                image = self.remove_noise(image)
            
            # Resize to target size
            image = self.resize_image(image)
            
            # Normalize pixel values
            image = self.normalize_image(image)
            
            return image
        except Exception as e:
            print(f"Error in preprocessing pipeline: {e}")
            return None
    
    def batch_preprocess(self, image_paths: list, enhance: bool = True) -> list:
        """Process multiple images"""
        processed_images = []
        
        for image_path in image_paths:
            processed_image = self.preprocess_for_model(image_path, enhance)
            if processed_image is not None:
                processed_images.append(processed_image)
        
        return processed_images
    
    def save_processed_image(self, image: np.ndarray, output_path: str) -> bool:
        """Save processed image"""
        try:
            # Convert back to uint8 if needed
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            pil_image.save(output_path)
            return True
        except Exception as e:
            print(f"Error saving processed image: {e}")
            return False
    
    def get_image_info(self, image_path: str) -> dict:
        """Get basic information about the image"""
        try:
            image = Image.open(image_path)
            return {
                'filename': os.path.basename(image_path),
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'file_size': os.path.getsize(image_path)
            }
        except Exception as e:
            print(f"Error getting image info: {e}")
            return {}
    
    def validate_image(self, image_path: str) -> tuple[bool, str]:
        """Validate if image is suitable for OCT analysis"""
        try:
            if not os.path.exists(image_path):
                return False, "File does not exist"
            
            # Check file size (should be reasonable)
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                return False, "File too large"
            
            if file_size < 1024:  # 1KB minimum
                return False, "File too small"
            
            # Try to load image
            image = self.load_image(image_path)
            if image is None:
                return False, "Cannot load image"
            
            # Check image dimensions
            height, width = image.shape[:2]
            if height < 64 or width < 64:
                return False, "Image too small"
            
            if height > 4096 or width > 4096:
                return False, "Image too large"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

def create_augmented_dataset(input_dir: str, output_dir: str, augment_factor: int = 3):
    """Create augmented dataset from existing images"""
    processor = OCTImageProcessor()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                input_path = os.path.join(root, file)
                
                # Create corresponding output directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                # Load and process original image
                original_image = processor.load_image(input_path)
                if original_image is None:
                    continue
                
                # Save original
                output_path = os.path.join(output_subdir, file)
                processor.save_processed_image(original_image, output_path)
                
                # Create augmented versions
                for i in range(augment_factor):
                    augmented = original_image.copy()
                    
                    # Apply random augmentations
                    if i % 3 == 0:
                        augmented = processor.enhance_contrast(augmented, factor=1.5)
                    elif i % 3 == 1:
                        augmented = processor.enhance_brightness(augmented, factor=1.2)
                    else:
                        augmented = processor.apply_gaussian_blur(augmented, kernel_size=1)
                    
                    # Save augmented version
                    name, ext = os.path.splitext(file)
                    aug_filename = f"{name}_aug_{i+1}{ext}"
                    aug_output_path = os.path.join(output_subdir, aug_filename)
                    processor.save_processed_image(augmented, aug_output_path)

if __name__ == "__main__":
    # Example usage
    processor = OCTImageProcessor()
    
    # Test preprocessing
    test_image_path = "uploads/test_image.jpg"
    if os.path.exists(test_image_path):
        processed = processor.preprocess_for_model(test_image_path)
        if processed is not None:
            print(f"Processed image shape: {processed.shape}")
            print(f"Processed image dtype: {processed.dtype}")
            print(f"Pixel value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Get image info
        info = processor.get_image_info(test_image_path)
        print(f"Image info: {info}")
        
        # Validate image
        is_valid, message = processor.validate_image(test_image_path)
        print(f"Image validation: {is_valid} - {message}")
