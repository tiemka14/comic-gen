#!/usr/bin/env python3
"""
Dataset Structure Checker
Diagnostic script to verify dataset directory structure for Kohya SS training.
"""

import os
import sys
from pathlib import Path


def check_dataset_structure(comic_name: str):
    """Check if the dataset structure is correct for Kohya SS training"""
    print(f"🔍 Checking dataset structure for: {comic_name}")
    print("=" * 50)
    
    # Expected paths
    dataset_path = f"./dataset/{comic_name}"
    train_dir = f"{dataset_path}/train"
    val_dir = f"{dataset_path}/validation"
    
    # Check main dataset directory
    print(f"📁 Checking dataset directory: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        print("   Run dataset preparation first: python scripts/train_lora.py --comic_name your_comic --prepare_only")
        return False
    else:
        print(f"✅ Dataset directory exists: {dataset_path}")
    
    # Check train directory
    print(f"\n📁 Checking training directory: {train_dir}")
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return False
    else:
        print(f"✅ Training directory exists: {train_dir}")
    
    # Check validation directory
    print(f"\n📁 Checking validation directory: {val_dir}")
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return False
    else:
        print(f"✅ Validation directory exists: {val_dir}")
    
    # Check for images in train directory
    print(f"\n🖼️ Checking training images...")
    train_images = [f for f in os.listdir(train_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not train_images:
        print(f"❌ No images found in training directory")
        return False
    else:
        print(f"✅ Found {len(train_images)} training images")
        print(f"   Sample images: {train_images[:3]}")
    
    # Check for images in validation directory
    print(f"\n🖼️ Checking validation images...")
    val_images = [f for f in os.listdir(val_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not val_images:
        print(f"❌ No images found in validation directory")
        return False
    else:
        print(f"✅ Found {len(val_images)} validation images")
        print(f"   Sample images: {val_images[:3]}")
    
    # Check for caption files
    print(f"\n📝 Checking caption files...")
    train_captions = [f for f in os.listdir(train_dir) if f.endswith('.txt')]
    val_captions = [f for f in os.listdir(val_dir) if f.endswith('.txt')]
    
    print(f"   Training captions: {len(train_captions)}")
    print(f"   Validation captions: {len(val_captions)}")
    
    # Check if captions match images
    train_image_names = {Path(f).stem for f in train_images}
    train_caption_names = {Path(f).stem for f in train_captions}
    
    missing_captions = train_image_names - train_caption_names
    if missing_captions:
        print(f"⚠️ Missing captions for training images: {list(missing_captions)[:5]}")
    
    # Show directory structure
    print(f"\n📂 Directory structure:")
    print(f"   {dataset_path}/")
    print(f"   ├── train/ ({len(train_images)} images, {len(train_captions)} captions)")
    print(f"   └── validation/ ({len(val_images)} images, {len(val_captions)} captions)")
    
    # Check Kohya SS compatibility
    print(f"\n🔧 Kohya SS compatibility check:")
    print(f"   Expected train_data_dir: {dataset_path}")
    print(f"   This should contain 'train' and 'validation' subdirectories")
    
    if len(train_images) > 0 and len(val_images) > 0:
        print(f"✅ Dataset structure is correct for Kohya SS training!")
        print(f"   Total images: {len(train_images) + len(val_images)}")
        print(f"   Ready for training!")
        return True
    else:
        print(f"❌ Dataset structure is incomplete")
        return False


def check_input_images(comic_name: str):
    """Check if input images directory exists and contains images"""
    print(f"\n🖼️ Checking input images for: {comic_name}")
    print("=" * 30)
    
    input_dir = "input_images"
    print(f"📁 Checking input directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("   Please create this directory and upload your comic images")
        return False
    
    images = [f for f in os.listdir(input_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        print("   Please upload your comic images to this directory")
        return False
    
    print(f"✅ Found {len(images)} images in {input_dir}")
    print(f"   Sample images: {images[:5]}")
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python check_dataset.py <comic_name>")
        print("Example: python check_dataset.py my_comic")
        return
    
    comic_name = sys.argv[1]
    
    print("🔍 Comic-Gen Dataset Structure Checker")
    print("=" * 50)
    
    # Check input images
    has_input = check_input_images(comic_name)
    
    # Check dataset structure
    has_dataset = check_dataset_structure(comic_name)
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    
    if not has_input:
        print("❌ No input images found")
        print("   → Create 'input_images' directory")
        print("   → Upload your comic images")
        print("   → Run: python scripts/train_lora.py --comic_name your_comic --prepare_only")
    
    elif not has_dataset:
        print("❌ Dataset not prepared")
        print("   → Run: python scripts/train_lora.py --comic_name your_comic --prepare_only")
    
    else:
        print("✅ Dataset is ready for training!")
        print("   → Run: python scripts/train_lora.py --comic_name your_comic")
    
    print("\n💡 Tips:")
    print("   - Use 20-100 high-quality images")
    print("   - Ensure consistent art style")
    print("   - Include various poses and expressions")
    print("   - Images will be automatically resized to 512x512")


if __name__ == "__main__":
    main() 