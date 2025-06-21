#!/usr/bin/env python3
"""
Fix Dataset Structure for Kohya SS
Converts existing dataset structure to the correct format with repetition counts.
"""

import os
import shutil
from pathlib import Path


def fix_dataset_structure(comic_name: str):
    """Fix the dataset structure to match Kohya SS requirements"""
    print(f"🔧 Fixing dataset structure for: {comic_name}")
    print("=" * 50)
    
    dataset_path = f"./dataset/{comic_name}"
    train_dir = f"{dataset_path}/train"
    val_dir = f"{dataset_path}/validation"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run dataset preparation first: python scripts/train_lora.py --comic_name your_comic --prepare_only")
        return False
    
    # Create repetition directories
    train_repeat_dir = os.path.join(train_dir, f"10_{comic_name}")
    val_repeat_dir = os.path.join(val_dir, f"10_{comic_name}")
    
    print(f"📁 Creating repetition directories...")
    os.makedirs(train_repeat_dir, exist_ok=True)
    os.makedirs(val_repeat_dir, exist_ok=True)
    
    # Move files from train directory to repetition directory
    if os.path.exists(train_dir):
        print(f"🔄 Moving training files...")
        moved_count = 0
        
        for item in os.listdir(train_dir):
            item_path = os.path.join(train_dir, item)
            
            # Skip the repetition directory itself
            if item == f"10_{comic_name}":
                continue
            
            # Move files to repetition directory
            if os.path.isfile(item_path):
                dest_path = os.path.join(train_repeat_dir, item)
                shutil.move(item_path, dest_path)
                moved_count += 1
        
        print(f"   Moved {moved_count} training files")
    
    # Move files from validation directory to repetition directory
    if os.path.exists(val_dir):
        print(f"🔄 Moving validation files...")
        moved_count = 0
        
        for item in os.listdir(val_dir):
            item_path = os.path.join(val_dir, item)
            
            # Skip the repetition directory itself
            if item == f"10_{comic_name}":
                continue
            
            # Move files to repetition directory
            if os.path.isfile(item_path):
                dest_path = os.path.join(val_repeat_dir, item)
                shutil.move(item_path, dest_path)
                moved_count += 1
        
        print(f"   Moved {moved_count} validation files")
    
    # Verify the new structure
    print(f"\n🔍 Verifying new structure...")
    
    train_images = [f for f in os.listdir(train_repeat_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    val_images = [f for f in os.listdir(val_repeat_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    train_captions = [f for f in os.listdir(train_repeat_dir) if f.endswith('.txt')]
    val_captions = [f for f in os.listdir(val_repeat_dir) if f.endswith('.txt')]
    
    print(f"✅ Structure fixed!")
    print(f"   Training: {len(train_images)} images, {len(train_captions)} captions")
    print(f"   Validation: {len(val_images)} images, {len(val_captions)} captions")
    
    print(f"\n📂 New directory structure:")
    print(f"   {dataset_path}/")
    print(f"   ├── train/")
    print(f"   │   └── 10_{comic_name}/ ({len(train_images)} images)")
    print(f"   └── validation/")
    print(f"       └── 10_{comic_name}/ ({len(val_images)} images)")
    
    print(f"\n🚀 Ready for training!")
    print(f"   Run: python scripts/train_lora.py --comic_name {comic_name}")
    
    return True


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_dataset_structure.py <comic_name>")
        print("Example: python fix_dataset_structure.py my_comic")
        return
    
    comic_name = sys.argv[1]
    fix_dataset_structure(comic_name)


if __name__ == "__main__":
    main() 