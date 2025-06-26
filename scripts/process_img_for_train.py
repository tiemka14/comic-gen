from PIL import Image
import os
import shutil

def process_image(image_path: str, target_size: int = 512) -> Image.Image:
    """
    Process and resize image for training
    
    Args:
        image_path: Path to input image
        target_size: Target size for training
        
    Returns:
        Processed PIL Image
    """
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize maintaining aspect ratio
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Create square image with padding if necessary
        new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        new_img.paste(img, ((target_size - img.width) // 2, (target_size - img.height) // 2))
        
        return new_img

if __name__ == "__main__":
    input_folder = "input_to_annotate"
    train_dir = f"dataset/train"
    os.makedirs(train_dir, exist_ok=True)

    print(f"Processing images from {input_folder} and saving to {train_dir}")
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
            original_path = os.path.join(input_folder, filename)

            processed_image = process_image(original_path)
            processed_image.save(f"{train_dir}/{filename}")

            # copy the caption file to the train directory
            caption_path = os.path.join(input_folder, f"{filename.replace('.jpg', '.txt')}")
            print(f"Caption path: {caption_path}")
            shutil.copy(caption_path, f"{train_dir}/{filename.replace('.jpg', '.txt')}")

            print(f"Processed {filename} and copied caption file")